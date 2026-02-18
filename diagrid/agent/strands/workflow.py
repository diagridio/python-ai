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

"""Dapr Agent Workflow - Run Strands Agents as Dapr Workflows.

This module provides the DaprAgentWorkflow class that wraps a Strands agent
and executes it as a durable Dapr workflow.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, TypeVar

from strands import Agent
from strands.agent import AgentResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class WorkflowInput:
    """Input for the agent workflow."""

    prompt: str
    conversation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowOutput:
    """Output from the agent workflow."""

    result: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DaprAgentWorkflow:
    """Wraps a Strands agent to run as a Dapr Workflow.

    Example:
        ```python
        from dapr.ext.workflow import WorkflowRuntime, DaprWorkflowClient
        from strands import Agent

        agent = Agent(model="us.amazon.nova-pro-v1:0", tools=[my_tool])
        dapr_workflow = DaprAgentWorkflow(agent)

        # Register with Dapr workflow runtime
        workflow_runtime = WorkflowRuntime()
        dapr_workflow.register(workflow_runtime)
        workflow_runtime.start()

        # Run the agent as a workflow
        client = DaprWorkflowClient()
        result = dapr_workflow.run(client, "Hello!")
        ```
    """

    def __init__(
        self,
        agent: Agent,
        workflow_name: str = "strands_agent_workflow",
    ) -> None:
        self.agent = agent
        self.workflow_name = workflow_name
        self._workflow_runtime: Any = None
        self._workflow_func: Any = None
        self._activity_name = f"{workflow_name}_run_agent"

    def register(self, workflow_runtime: Any) -> None:
        """Register the agent workflow and activities with Dapr.

        Args:
            workflow_runtime: The Dapr WorkflowRuntime instance
        """
        self._workflow_runtime = workflow_runtime

        # Store reference to self for use in closures
        agent = self.agent
        activity_name = self._activity_name
        workflow_name = self.workflow_name

        # Define the activity that runs the agent
        def run_agent_activity(ctx: Any, input_data: dict) -> dict:  # type: ignore[type-arg]
            """Activity that executes the Strands agent."""
            print(
                f"[ACTIVITY] run_agent_activity called with: {input_data}",
                flush=True,
            )

            prompt = input_data.get("prompt", "")
            conversation_id = input_data.get("conversation_id", str(uuid.uuid4()))

            print(
                f"[ACTIVITY] Running agent with prompt: {prompt[:100]}...",
                flush=True,
            )

            try:
                # Run the agent synchronously (activity context)
                result = asyncio.run(agent.invoke_async(prompt))

                result_str = str(result)
                print(
                    f"[ACTIVITY] Agent completed: {result_str[:200]}...",
                    flush=True,
                )

                # Extract tool calls
                tool_calls: list[dict[str, Any]] = []
                for message in agent.messages:
                    if message.get("role") == "assistant":
                        for content in message.get("content", []):
                            if "toolUse" in content:
                                tool_use = content["toolUse"]
                                tool_calls.append(
                                    {
                                        "tool_name": tool_use.get("name"),
                                        "tool_use_id": tool_use.get("toolUseId"),
                                        "input": tool_use.get("input"),
                                    }
                                )

                print(
                    f"[ACTIVITY] Returning result with {len(tool_calls)} tool calls",
                    flush=True,
                )
                return {
                    "result": result_str,
                    "tool_calls": tool_calls,
                    "conversation_id": conversation_id,
                }

            except Exception as e:
                print(f"[ACTIVITY] Error: {e}", flush=True)
                import traceback

                traceback.print_exc()
                return {
                    "result": f"Error: {str(e)}",
                    "tool_calls": [],
                    "conversation_id": conversation_id,
                    "error": str(e),
                }

        # Define the workflow (generator function with yield)
        def agent_workflow(ctx: Any, input_data: dict) -> Generator[Any, Any, dict]:  # type: ignore[type-arg]
            """Dapr workflow that orchestrates the agent."""
            print(
                f"[WORKFLOW] Starting workflow with input: {input_data}",
                flush=True,
            )

            # Call the agent activity and yield to wait for completion
            result = yield ctx.call_activity(run_agent_activity, input=input_data)

            print(
                f"[WORKFLOW] Activity completed with result: {str(result)[:200]}",
                flush=True,
            )

            return result  # type: ignore[return-value]

        # Register with Dapr
        workflow_runtime.register_workflow(agent_workflow, name=workflow_name)
        workflow_runtime.register_activity(run_agent_activity, name=activity_name)

        # Store reference for scheduling
        self._workflow_func = agent_workflow

        logger.info(
            "workflow=%s activity=%s | registered with Dapr",
            workflow_name,
            activity_name,
        )
        print(
            f"[REGISTER] Registered workflow '{workflow_name}' and activity '{activity_name}'",
            flush=True,
        )

    def start(
        self,
        workflow_client: Any,
        prompt: str,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        instance_id: str | None = None,
    ) -> str:
        """Start the agent workflow.

        Args:
            workflow_client: The DaprWorkflowClient instance
            prompt: The user prompt
            conversation_id: Optional conversation ID
            metadata: Optional metadata
            instance_id: Optional workflow instance ID

        Returns:
            The workflow instance ID
        """
        if instance_id is None:
            instance_id = f"{self.workflow_name}_{uuid.uuid4().hex[:8]}"

        input_data = {
            "prompt": prompt,
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "metadata": metadata or {},
        }

        print(f"[START] Starting workflow instance: {instance_id}", flush=True)

        workflow_client.schedule_new_workflow(
            workflow=self._workflow_func,
            input=input_data,
            instance_id=instance_id,
        )

        return instance_id

    def wait_for_completion(
        self,
        workflow_client: Any,
        instance_id: str,
        timeout_seconds: int = 120,
    ) -> WorkflowOutput:
        """Wait for workflow completion and return the result.

        Args:
            workflow_client: The DaprWorkflowClient instance
            instance_id: The workflow instance ID
            timeout_seconds: Timeout in seconds

        Returns:
            WorkflowOutput with the agent's response
        """
        import time

        start_time = time.time()

        while True:
            state = workflow_client.get_workflow_state(
                instance_id=instance_id,
            )

            status = str(state.runtime_status)
            print(
                f"[WAIT] Workflow status: {status} (raw: {state.runtime_status})",
                flush=True,
            )

            # Check for terminal states
            status_upper = status.upper()
            if (
                "COMPLETED" in status_upper
                or "FAILED" in status_upper
                or "TERMINATED" in status_upper
            ):
                print(
                    f"[WAIT] Workflow reached terminal state: {status}",
                    flush=True,
                )
                break

            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Workflow {instance_id} timed out")

            time.sleep(1)

        if "FAILED" in str(state.runtime_status).upper():
            raise RuntimeError("Workflow failed")

        # Try different attribute names for the output
        output = None
        for attr in ["serialized_output", "workflow_output", "output", "result"]:
            if hasattr(state, attr):
                output = getattr(state, attr)
                print(f"[WAIT] Found output in '{attr}': {output}", flush=True)
                break

        if output is None:
            print(f"[WAIT] No output found, state: {state}", flush=True)
            output = {}
        elif isinstance(output, str):
            output = json.loads(output)

        print(f"[WAIT] Parsed output: {output}", flush=True)

        return WorkflowOutput(
            result=output.get("result", "")
            if isinstance(output, dict)
            else str(output),
            tool_calls=output.get("tool_calls", []) if isinstance(output, dict) else [],
            conversation_id=output.get("conversation_id")
            if isinstance(output, dict)
            else None,
            metadata=output.get("metadata", {}) if isinstance(output, dict) else {},
        )

    def run(
        self,
        workflow_client: Any,
        prompt: str,
        conversation_id: str | None = None,
        timeout_seconds: int = 120,
    ) -> WorkflowOutput:
        """Start workflow and wait for completion.

        This is a convenience method that combines start() and wait_for_completion().

        Args:
            workflow_client: The DaprWorkflowClient instance
            prompt: The user prompt
            conversation_id: Optional conversation ID
            timeout_seconds: Timeout in seconds

        Returns:
            WorkflowOutput with the agent's response
        """
        instance_id = self.start(
            workflow_client=workflow_client,
            prompt=prompt,
            conversation_id=conversation_id,
        )

        return self.wait_for_completion(
            workflow_client=workflow_client,
            instance_id=instance_id,
            timeout_seconds=timeout_seconds,
        )


def dapr_agent_workflow(
    workflow_name: str = "strands_agent_workflow",
) -> Callable[[Callable[..., Agent]], Callable[..., DaprAgentWorkflow]]:
    """Decorator to create a Dapr workflow from an agent factory function.

    Example:
        ```python
        @dapr_agent_workflow(workflow_name="my_agent")
        def create_agent() -> Agent:
            return Agent(model="us.amazon.nova-pro-v1:0", tools=[my_tool])

        workflow = create_agent()
        workflow.register(workflow_runtime)
        ```
    """

    def decorator(
        agent_factory: Callable[..., Agent],
    ) -> Callable[..., DaprAgentWorkflow]:
        def wrapper(*args: Any, **kwargs: Any) -> DaprAgentWorkflow:
            agent = agent_factory(*args, **kwargs)
            return DaprAgentWorkflow(agent=agent, workflow_name=workflow_name)

        return wrapper

    return decorator
