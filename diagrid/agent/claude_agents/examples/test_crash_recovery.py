# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Test script to verify Dapr workflow crash recovery for Claude Agent SDK agents.

This test:
1. Defines an agent with three tools that must be called in sequence
2. Crashes the process during tool 2 execution (after tool 1 completes)
3. On restart, Dapr automatically resumes the workflow and completes it

Usage:
    # Clean up any previous test state first:
    rm -f /tmp/claude_agents_crash_test_state.json

    # First run (will crash during tool 2):
    dapr run --app-id claude-agents-crash-test --resources-path ./components -- python3 test_crash_recovery.py

    # Second run (Dapr auto-resumes and completes):
    dapr run --app-id claude-agents-crash-test --resources-path ./components -- python3 test_crash_recovery.py
"""

import asyncio
import json
import os
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, tool
from dapr.ext.workflow import WorkflowStatus

from diagrid.agent.claude_agents import DaprWorkflowAgentRunner


def log(msg: str):
    """Print with immediate flush."""
    print(msg, flush=True)


STATE_FILE = Path("/tmp/claude_agents_crash_test_state.json")
SESSION_ID = "crash-recovery-test"


def load_state() -> dict:
    """Load the test state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "run_count": 0,
        "tool1_executed": False,
        "tool2_executed": False,
        "tool3_executed": False,
        "workflow_scheduled": False,
        "workflow_id": None,
    }


def save_state(state: dict):
    """Save the test state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


state = load_state()
state["run_count"] += 1
save_state(state)

log(f"\n{'=' * 60}")
log(f"RUN #{state['run_count']}")
log(f"{'=' * 60}")
log(
    f"Previous state: tool1={state['tool1_executed']}, "
    f"tool2={state['tool2_executed']}, tool3={state['tool3_executed']}"
)
log(f"Workflow previously scheduled: {state['workflow_scheduled']}")
log(f"Saved workflow_id: {state.get('workflow_id')}")
log(f"{'=' * 60}\n")


@tool(
    "step_one", "Initialize data for the workflow. The first step.", {"input_data": str}
)
async def step_one(args):
    """First step: initialize data."""
    log(f"\n>>> TOOL 1 EXECUTING: input={args['input_data']}")
    state["tool1_executed"] = True
    save_state(state)
    log(">>> TOOL 1 COMPLETED SUCCESSFULLY")
    text = (
        f"Step 1 completed: Initialized with '{args['input_data']}'. Now call step_two."
    )
    return {"content": [{"type": "text", "text": text}]}


@tool("step_two", "Process the data from step one. The second step.", {"data": str})
async def step_two(args):
    """Second step: process the data. Crashes on first run."""
    log(f"\n>>> TOOL 2 EXECUTING: data={args['data']}")

    if state["run_count"] == 1:
        log(">>> TOOL 2: SIMULATING CRASH!")
        log(">>> The process will now terminate...")
        log(">>> Run the program again to test recovery.\n")
        os._exit(1)

    state["tool2_executed"] = True
    save_state(state)
    log(">>> TOOL 2 COMPLETED SUCCESSFULLY")
    text = f"Step 2 completed: Processed '{args['data']}'. Now call step_three."
    return {"content": [{"type": "text", "text": text}]}


@tool(
    "step_three",
    "Finalize and return the results. The third and final step.",
    {"processed_data": str},
)
async def step_three(args):
    """Third step: finalize the result."""
    log(f"\n>>> TOOL 3 EXECUTING: processed_data={args['processed_data']}")
    state["tool3_executed"] = True
    save_state(state)
    log(">>> TOOL 3 COMPLETED SUCCESSFULLY")
    text = (
        f"Step 3 completed: Final result based on '{args['processed_data']}'. "
        "All steps done!"
    )
    return {"content": [{"type": "text", "text": text}]}


SYSTEM_PROMPT = (
    "You are a sequential task processor that MUST call tools in a specific order. "
    "You MUST call all three tools in sequence: "
    "1. First call 'step_one' with the input. "
    "2. Then call 'step_two' with the output from step 1. "
    "3. Finally call 'step_three' with the output from step 2. "
    "Do NOT skip any steps. Each tool must be called exactly once in order."
)

PROMPT = (
    'Process the input "test_data_123" through all three steps.\n\n'
    "You MUST:\n"
    '1. Call step_one with "test_data_123"\n'
    "2. Call step_two with the result from step 1\n"
    "3. Call step_three with the result from step 2\n\n"
    "Call each tool exactly once in sequence."
)


async def main():
    """Run the crash recovery test."""

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
    )

    runner = DaprWorkflowAgentRunner(
        name="crash-recovery-test",
        options=options,
        tools=[step_one, step_two, step_three],
    )

    try:
        runner.start()
        log("Agent runtime started")
        await asyncio.sleep(1)

        if not state["workflow_scheduled"]:
            log("Scheduling new workflow...")
            async for event in runner.run_async(
                user_message=PROMPT,
                session_id=SESSION_ID,
            ):
                event_type = event["type"]
                log(f"Event: {event_type}")
                if event_type == "workflow_started":
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


async def poll_for_completion(runner: DaprWorkflowAgentRunner, workflow_id: str):
    """Poll an existing workflow until it completes."""
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
                print_completion(
                    {
                        "final_response": output_dict.get("final_response", ""),
                        "iterations": output_dict.get("iterations", 0),
                        "status": output_dict.get("status"),
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
    log(f"Final response:\n{event.get('final_response')}")
    log(f"Iterations: {event.get('iterations')}")
    log(f"Status: {event.get('status')}")

    final_state = load_state()
    log(f"\n{'=' * 60}")
    log("VERIFICATION:")
    log(f"{'=' * 60}")
    log(f"Tool 1 executed: {final_state['tool1_executed']}")
    log(f"Tool 2 executed: {final_state['tool2_executed']}")
    log(f"Tool 3 executed: {final_state['tool3_executed']}")
    log(f"Total runs: {final_state['run_count']}")

    if final_state["run_count"] >= 2 and all(
        [
            final_state["tool1_executed"],
            final_state["tool2_executed"],
            final_state["tool3_executed"],
        ]
    ):
        log("\n>>> TEST PASSED: Crash recovery worked!")
        log(">>> Workflow resumed after crash and completed all tools.")


if __name__ == "__main__":
    asyncio.run(main())
