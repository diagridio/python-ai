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

"""
Test script to verify Dapr workflow activity retry behavior.

This test:
1. Creates a graph with 3 nodes in sequence
2. Node 2 raises a ConnectionError on the first 2 attempts, then succeeds
3. Dapr's retry policy (3 attempts) retries the activity automatically
4. The workflow completes successfully without re-executing node 1

Usage:
    dapr run --app-id langgraph-retry-test --resources-path ./components -- python3 test_retry.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

# State file to track execution attempts
STATE_FILE = Path("/tmp/langgraph_retry_test_state.json")
THREAD_ID = "retry-test"


def log(msg: str):
    """Print with immediate flush."""
    print(msg, flush=True)


def load_state() -> dict:
    """Load the test state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "node1_count": 0,
        "node2_count": 0,
        "node3_count": 0,
    }


def save_state(state: dict):
    """Save the test state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# Global state for tracking attempts
attempt_state = load_state()


# Define the graph state
class GraphState(TypedDict):
    messages: List[str]
    step: int


# Define node functions
def node_one(graph_state: GraphState) -> dict:
    """First node - always succeeds."""
    attempt_state["node1_count"] += 1
    save_state(attempt_state)
    log(f"\n>>> NODE 1 EXECUTED (attempt #{attempt_state['node1_count']})")
    return {
        "messages": graph_state["messages"] + ["Node 1 completed"],
        "step": graph_state["step"] + 1,
    }


def node_two(graph_state: GraphState) -> dict:
    """Second node - fails on first 2 attempts, succeeds on 3rd."""
    attempt_state["node2_count"] += 1
    save_state(attempt_state)
    count = attempt_state["node2_count"]
    log(f"\n>>> NODE 2 EXECUTING (attempt #{count})")

    if count <= 2:
        log(f">>> NODE 2: RAISING ConnectionError (attempt {count}/3)")
        raise ConnectionError(f"Chaos: LLM call failed (attempt {count})")

    log(">>> NODE 2 COMPLETED SUCCESSFULLY on attempt #3")
    return {
        "messages": graph_state["messages"] + ["Node 2 completed after retries"],
        "step": graph_state["step"] + 1,
    }


def node_three(graph_state: GraphState) -> dict:
    """Third node - always succeeds."""
    attempt_state["node3_count"] += 1
    save_state(attempt_state)
    log(f"\n>>> NODE 3 EXECUTED (attempt #{attempt_state['node3_count']})")
    return {
        "messages": graph_state["messages"] + ["Node 3 completed"],
        "step": graph_state["step"] + 1,
    }


def build_graph() -> StateGraph:
    """Build a simple 3-node graph."""
    graph = StateGraph(GraphState)

    graph.add_node("node_one", node_one)
    graph.add_node("node_two", node_two)
    graph.add_node("node_three", node_three)

    graph.add_edge(START, "node_one")
    graph.add_edge("node_one", "node_two")
    graph.add_edge("node_two", "node_three")
    graph.add_edge("node_three", END)

    return graph


async def main():
    """Run the retry test."""
    # Reset state
    save_state({"node1_count": 0, "node2_count": 0, "node3_count": 0})

    log(f"\n{'=' * 60}")
    log("RETRY TEST")
    log(f"{'=' * 60}")
    log("Graph: START -> node_one -> node_two -> node_three -> END")
    log("node_two will fail on attempts 1 and 2, succeed on attempt 3")
    log("Dapr retry policy: max 3 attempts, exponential backoff")
    log(f"{'=' * 60}\n")

    graph = build_graph()
    compiled = graph.compile()

    runner = DaprWorkflowGraphRunner(
        graph=compiled,
        max_steps=50,
        name="retry_test",
    )

    try:
        runner.start()
        log("Workflow runtime started")
        await asyncio.sleep(1)

        input_state = {"messages": ["Start"], "step": 0}

        async for event in runner.run_async(
            input=input_state,
            thread_id=THREAD_ID,
        ):
            event_type = event["type"]

            if event_type == "workflow_started":
                log(f"Workflow started: {event.get('workflow_id')}")
            elif event_type == "workflow_status_changed":
                log(f"Status: {event.get('status')}")
            elif event_type == "workflow_completed":
                print_completion(event)
                break
            elif event_type == "workflow_failed":
                log(f"\nWorkflow FAILED: {event.get('error')}")
                print_verification()
                break

    except KeyboardInterrupt:
        log("\nInterrupted by user")
    finally:
        runner.shutdown()
        log("Workflow runtime stopped")


def print_completion(event: dict):
    """Print completion summary and verification."""
    log(f"\n{'=' * 60}")
    log("WORKFLOW COMPLETED!")
    log(f"{'=' * 60}")
    log(f"Output: {event.get('output')}")
    log(f"Steps: {event.get('steps')}")
    print_verification()


def print_verification():
    """Print the verification results."""
    final_state = load_state()
    log(f"\n{'=' * 60}")
    log("VERIFICATION:")
    log(f"{'=' * 60}")
    log(f"Node 1 executions: {final_state['node1_count']} (expected: 1)")
    log(f"Node 2 executions: {final_state['node2_count']} (expected: 3 = 2 failures + 1 success)")
    log(f"Node 3 executions: {final_state['node3_count']} (expected: 1)")

    if (
        final_state["node1_count"] == 1
        and final_state["node2_count"] == 3
        and final_state["node3_count"] == 1
    ):
        log("\n>>> TEST PASSED: Retry worked!")
        log(">>> Node 2 was retried by Dapr and succeeded on attempt 3.")
        log(">>> Node 1 was NOT re-executed (durable checkpoint preserved).")
    elif final_state["node2_count"] < 3 and final_state["node3_count"] == 0:
        log("\n>>> TEST FAILED: Node 2 errors were swallowed instead of retried.")
        log(">>> The workflow completed without Dapr retrying the failed activity.")
    else:
        log(f"\n>>> UNEXPECTED: Check execution counts above.")


if __name__ == "__main__":
    asyncio.run(main())
