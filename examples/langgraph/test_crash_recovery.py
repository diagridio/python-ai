# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Test script to verify Dapr workflow crash recovery for LangGraph.

This test:
1. Creates a graph with 3 nodes that execute in sequence
2. Crashes the process during node 2 execution (after node 1 completes)
3. On restart, Dapr automatically resumes the workflow and completes it

Usage:
    # Clean up any previous test state first:
    rm -f /tmp/langgraph_crash_test_state.json

    # First run (will crash during node 2):
    dapr run --app-id langgraph-crash-test --resources-path ./components -- python test_crash_recovery.py

    # Second run (Dapr auto-resumes and completes):
    dapr run --app-id langgraph-crash-test --resources-path ./components -- python test_crash_recovery.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from dapr.ext.workflow import WorkflowStatus

# State file to track execution across crashes
STATE_FILE = Path("/tmp/langgraph_crash_test_state.json")
THREAD_ID = "crash-recovery-test"


def log(msg: str):
    """Print with immediate flush."""
    print(msg, flush=True)


def load_state() -> dict:
    """Load the test state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "run_count": 0,
        "node1_executed": False,
        "node2_executed": False,
        "node3_executed": False,
        "workflow_scheduled": False,
        "workflow_id": None,  # Store the actual workflow ID
    }


def save_state(state: dict):
    """Save the test state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# Global state for this run
state = load_state()
state["run_count"] += 1
save_state(state)

log(f"\n{'=' * 60}")
log(f"RUN #{state['run_count']}")
log(f"{'=' * 60}")
log(
    f"Previous state: node1={state['node1_executed']}, "
    f"node2={state['node2_executed']}, node3={state['node3_executed']}"
)
log(f"Workflow previously scheduled: {state['workflow_scheduled']}")
log(f"Saved workflow_id: {state.get('workflow_id')}")
log(f"{'=' * 60}\n")


# Define the graph state
class GraphState(TypedDict):
    messages: List[str]
    step: int


# Define node functions
def node_one(graph_state: GraphState) -> dict:
    """First node - initialization."""
    log(f"\n>>> NODE 1 EXECUTING: step={graph_state['step']}")
    state["node1_executed"] = True
    save_state(state)
    log(">>> NODE 1 COMPLETED SUCCESSFULLY")
    return {
        "messages": graph_state["messages"] + ["Node 1 completed"],
        "step": graph_state["step"] + 1,
    }


def node_two(graph_state: GraphState) -> dict:
    """Second node - processing."""
    log(f"\n>>> NODE 2 EXECUTING: step={graph_state['step']}")

    # On first run, crash during node 2 (after node 1 completed)
    if state["run_count"] == 1:
        log(">>> NODE 2: SIMULATING CRASH!")
        log(">>> The process will now terminate...")
        log(">>> Run the program again to test recovery.\n")
        os._exit(1)

    state["node2_executed"] = True
    save_state(state)
    log(">>> NODE 2 COMPLETED SUCCESSFULLY")
    return {
        "messages": graph_state["messages"] + ["Node 2 completed"],
        "step": graph_state["step"] + 1,
    }


def node_three(graph_state: GraphState) -> dict:
    """Third node - finalization."""
    log(f"\n>>> NODE 3 EXECUTING: step={graph_state['step']}")
    state["node3_executed"] = True
    save_state(state)
    log(">>> NODE 3 COMPLETED SUCCESSFULLY")
    return {
        "messages": graph_state["messages"] + ["Node 3 completed"],
        "step": graph_state["step"] + 1,
    }


def build_graph() -> StateGraph:
    """Build a simple 3-node graph."""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("node_one", node_one)
    graph.add_node("node_two", node_two)
    graph.add_node("node_three", node_three)

    # Add edges: START -> node_one -> node_two -> node_three -> END
    graph.add_edge(START, "node_one")
    graph.add_edge("node_one", "node_two")
    graph.add_edge("node_two", "node_three")
    graph.add_edge("node_three", END)

    return graph


async def main():
    """Run the crash recovery test."""
    log("Building graph: START -> node_one -> node_two -> node_three -> END")

    graph = build_graph()
    compiled = graph.compile()

    runner = DaprWorkflowGraphRunner(
        graph=compiled,
        max_steps=50,
        name="crash_recovery_test",
    )

    try:
        runner.start()
        log("Workflow runtime started")
        await asyncio.sleep(1)

        # Only schedule a new workflow if we haven't already
        if not state["workflow_scheduled"]:
            log("Scheduling new workflow...")

            input_state = {"messages": ["Start"], "step": 0}
            log(f"Input state: {input_state}")

            async for event in runner.run_async(
                input=input_state,
                thread_id=THREAD_ID,
            ):
                event_type = event["type"]
                log(f"Event: {event_type}")

                if event_type == "workflow_started":
                    # Save the actual workflow_id for polling on restart
                    actual_workflow_id = event.get("workflow_id")
                    state["workflow_scheduled"] = True
                    state["workflow_id"] = actual_workflow_id
                    save_state(state)
                    log(f"Workflow started: {actual_workflow_id}")
                elif event_type == "workflow_status_changed":
                    log(f"Status: {event.get('status')}")
                elif event_type == "workflow_completed":
                    print_completion(event)
                    break
                elif event_type == "workflow_failed":
                    log(f"\nWorkflow FAILED: {event.get('error')}")
                    break
        else:
            # Workflow was already scheduled - poll using the saved workflow_id
            saved_workflow_id = state.get("workflow_id")
            log(
                f"Workflow already scheduled. Polling for completion: {saved_workflow_id}"
            )
            await poll_for_completion(runner, saved_workflow_id)

    except KeyboardInterrupt:
        log("\nInterrupted by user")
    finally:
        runner.shutdown()
        log("Workflow runtime stopped")


async def poll_for_completion(runner: DaprWorkflowGraphRunner, workflow_id: str):
    """Poll an existing workflow until it completes."""
    from diagrid.agent.langgraph.models import GraphWorkflowOutput

    if not workflow_id:
        log("No workflow_id saved - cannot poll!")
        return

    previous_status = None
    while True:
        await asyncio.sleep(1.0)
        workflow_state = runner._workflow_client.get_workflow_state(
            instance_id=workflow_id
        )

        if workflow_state is None:
            log("Workflow state not found!")
            break

        if workflow_state.runtime_status != previous_status:
            log(f"Workflow status: {workflow_state.runtime_status}")
            previous_status = workflow_state.runtime_status

        if workflow_state.runtime_status == WorkflowStatus.COMPLETED:
            output_data = workflow_state.serialized_output
            if output_data:
                output_dict = (
                    json.loads(output_data)
                    if isinstance(output_data, str)
                    else output_data
                )
                output = GraphWorkflowOutput.from_dict(output_dict)
                print_completion(
                    {
                        "output": output.output,
                        "steps": output.steps,
                        "status": output.status,
                    }
                )
            break
        elif workflow_state.runtime_status == WorkflowStatus.FAILED:
            log(f"\nWorkflow FAILED: {workflow_state.failure_details}")
            break
        elif workflow_state.runtime_status == WorkflowStatus.TERMINATED:
            log("\nWorkflow was TERMINATED")
            break


def print_completion(event: dict):
    """Print completion summary and verification."""
    log(f"\n{'=' * 60}")
    log("WORKFLOW COMPLETED!")
    log(f"{'=' * 60}")
    log(f"Output: {event.get('output')}")
    log(f"Steps: {event.get('steps')}")

    # Reload state to get latest
    final_state = load_state()
    log(f"\n{'=' * 60}")
    log("VERIFICATION:")
    log(f"{'=' * 60}")
    log(f"Node 1 executed: {final_state['node1_executed']}")
    log(f"Node 2 executed: {final_state['node2_executed']}")
    log(f"Node 3 executed: {final_state['node3_executed']}")
    log(f"Total runs: {final_state['run_count']}")

    if final_state["run_count"] >= 2 and all(
        [
            final_state["node1_executed"],
            final_state["node2_executed"],
            final_state["node3_executed"],
        ]
    ):
        log("\n>>> TEST PASSED: Crash recovery worked!")
        log(">>> Workflow resumed after crash and completed all nodes.")


if __name__ == "__main__":
    asyncio.run(main())
