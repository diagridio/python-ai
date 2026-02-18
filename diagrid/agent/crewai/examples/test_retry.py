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
Test script to verify Dapr workflow activity retry behavior for CrewAI agents.

This test:
1. Creates an agent with 3 tools that the LLM calls in sequence
2. Tool 2 raises a ConnectionError on the first 2 attempts, then succeeds
3. Dapr's retry policy (3 attempts) retries the activity automatically
4. The workflow completes successfully - tool 1 is NOT re-executed

Usage:
    rm -f /tmp/crewai_retry_test_state.json
    dapr run --app-id crewai-retry-test --resources-path ./components -- python3 test_retry.py
"""

import asyncio
import json
from pathlib import Path

from crewai import Agent, Task
from crewai.tools import tool

from diagrid.agent.crewai import DaprWorkflowAgentRunner

# State file to track execution attempts
STATE_FILE = Path("/tmp/crewai_retry_test_state.json")
SESSION_ID = "retry-test"


def log(msg: str):
    """Print with immediate flush."""
    print(msg, flush=True)


def load_state() -> dict:
    """Load the test state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "tool1_count": 0,
        "tool2_count": 0,
        "tool3_count": 0,
    }


def save_state(state: dict):
    """Save the test state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# Global state for tracking attempts
attempt_state = load_state()


@tool("Step one - Initialize data")
def tool_step_one(input_data: str) -> str:
    """Initialize data for the workflow. This is the first step."""
    attempt_state["tool1_count"] += 1
    save_state(attempt_state)
    log(f"\n>>> TOOL 1 EXECUTED (attempt #{attempt_state['tool1_count']})")
    return f"Step 1 completed: Initialized with '{input_data}'. Now call step_two___process_data."


@tool("Step two - Process data")
def tool_step_two(data: str) -> str:
    """Process the data from step one. This is the second step."""
    attempt_state["tool2_count"] += 1
    save_state(attempt_state)
    count = attempt_state["tool2_count"]
    log(f"\n>>> TOOL 2 EXECUTING (attempt #{count})")

    if count <= 2:
        log(f">>> TOOL 2: RAISING ConnectionError (attempt {count}/3)")
        raise ConnectionError(f"Chaos: LLM call failed (attempt {count})")

    log(">>> TOOL 2 COMPLETED SUCCESSFULLY on attempt #3")
    return (
        f"Step 2 completed: Processed '{data}'. Now call step_three___finalize_results."
    )


@tool("Step three - Finalize results")
def tool_step_three(processed_data: str) -> str:
    """Finalize and return the results. This is the third and final step."""
    attempt_state["tool3_count"] += 1
    save_state(attempt_state)
    log(f"\n>>> TOOL 3 EXECUTED (attempt #{attempt_state['tool3_count']})")
    return (
        f"Step 3 completed: Final result based on '{processed_data}'. All steps done!"
    )


# Create the agent
agent = Agent(
    role="Sequential Task Processor",
    goal="Execute exactly three tools in sequence: step_one, step_two, step_three",
    backstory="""You are a sequential task processor that MUST call tools in a specific order.
    You MUST call all three tools in sequence:
    1. First call 'step_one___initialize_data' with the input
    2. Then call 'step_two___process_data' with the output from step 1
    3. Finally call 'step_three___finalize_results' with the output from step 2

    Do NOT skip any steps. Each tool must be called exactly once in order.""",
    tools=[tool_step_one, tool_step_two, tool_step_three],
    llm="openai/gpt-4o-mini",
    verbose=True,
)

# Create a task
task = Task(
    description="""Process the input "test_data_123" through all three steps.

    You MUST:
    1. Call step_one___initialize_data with "test_data_123"
    2. Call step_two___process_data with the result from step 1
    3. Call step_three___finalize_results with the result from step 2

    Call each tool exactly once in sequence.""",
    expected_output="A summary confirming all three steps completed.",
    agent=agent,
)


async def main():
    """Run the retry test."""
    # Reset state
    save_state({"tool1_count": 0, "tool2_count": 0, "tool3_count": 0})

    log(f"\n{'=' * 60}")
    log("RETRY TEST - CrewAI")
    log(f"{'=' * 60}")
    log("Tool 2 will fail on attempts 1 and 2, succeed on attempt 3")
    log("Dapr retry policy: max 3 attempts, exponential backoff")
    log(f"{'=' * 60}\n")

    runner = DaprWorkflowAgentRunner(
        agent=agent,
        max_iterations=10,
    )

    try:
        runner.start()
        log("Workflow runtime started")
        await asyncio.sleep(1)

        async for event in runner.run_async(
            task=task,
            session_id=SESSION_ID,
        ):
            event_type = event["type"]
            log(f"Event: {event_type}")

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
    """Print completion summary."""
    log(f"\n{'=' * 60}")
    log("WORKFLOW COMPLETED!")
    log(f"{'=' * 60}")
    log(f"Final Response:\n{event.get('final_response')}")
    log(f"Iterations: {event.get('iterations')}")
    print_verification()


def print_verification():
    """Print the verification results."""
    final_state = load_state()
    log(f"\n{'=' * 60}")
    log("VERIFICATION:")
    log(f"{'=' * 60}")
    log(f"Tool 1 executions: {final_state['tool1_count']} (expected: 1)")
    log(
        f"Tool 2 executions: {final_state['tool2_count']} (expected: 3 = 2 failures + 1 success)"
    )
    log(f"Tool 3 executions: {final_state['tool3_count']} (expected: 1)")

    if (
        final_state["tool1_count"] == 1
        and final_state["tool2_count"] == 3
        and final_state["tool3_count"] == 1
    ):
        log("\n>>> TEST PASSED: Retry worked!")
        log(">>> Tool 2 was retried by Dapr and succeeded on attempt 3.")
        log(">>> Tool 1 was NOT re-executed (durable checkpoint preserved).")
    elif final_state["tool2_count"] < 3 and final_state["tool3_count"] == 0:
        log("\n>>> TEST FAILED: Tool 2 errors were swallowed instead of retried.")
        log(">>> The workflow completed without Dapr retrying the failed activity.")
    else:
        log("\n>>> UNEXPECTED: Check execution counts above.")


if __name__ == "__main__":
    asyncio.run(main())
