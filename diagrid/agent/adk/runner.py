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

import logging
import uuid
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING

from dapr.ext.workflow import WorkflowRuntime, DaprWorkflowClient, WorkflowStatus

from .models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    Message,
    MessageRole,
    ToolDefinition,
)
from .workflow import (
    adk_agent_workflow,
    call_llm_activity,
    execute_tool_activity,
    register_tool,
    clear_tool_registry,
)

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent

logger = logging.getLogger(__name__)


class DaprWorkflowAgentRunner:
    """Runner that executes Google ADK agents as Dapr Workflows.

    This runner wraps an ADK LlmAgent and executes it using Dapr Workflows,
    making each tool execution a durable activity. This provides:

    - Fault tolerance: Agents automatically resume from the last successful activity
    - Durability: Agent state persists and survives process restarts
    - Observability: Full visibility into agent execution through Dapr's workflow APIs

    Example:
        ```python
        from google.adk.agents import LlmAgent
        from google.adk.tools import FunctionTool
        from diagrid.agent.adk import DaprWorkflowAgentRunner

        # Define your ADK agent
        agent = LlmAgent(
            name="my_agent",
            model="gemini-2.0-flash",
            tools=[my_tool],
        )

        # Create runner and start the workflow runtime
        runner = DaprWorkflowAgentRunner(agent=agent)
        runner.start()

        # Run the agent
        async for event in runner.run_async(
            user_message="Hello!",
            session_id="session-123",
        ):
            print(event)

        # Shutdown when done
        runner.shutdown()
        ```

    Attributes:
        agent: The ADK LlmAgent to execute
        workflow_runtime: The Dapr WorkflowRuntime instance
        workflow_client: The Dapr WorkflowClient for managing workflows
    """

    def __init__(
        self,
        agent: "LlmAgent",
        *,
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_iterations: int = 100,
    ):
        """Initialize the runner.

        Args:
            agent: The ADK LlmAgent to execute
            host: Dapr sidecar host (default: localhost)
            port: Dapr sidecar port (default: 50001)
            max_iterations: Maximum number of LLM call iterations (default: 100)
        """
        self._agent = agent
        self._max_iterations = max_iterations
        self._host = host
        self._port = port

        # Create workflow runtime
        self._workflow_runtime = WorkflowRuntime(host=host, port=port)

        # Register workflow and activities
        self._workflow_runtime.register_workflow(
            adk_agent_workflow, name="adk_agent_workflow"
        )
        self._workflow_runtime.register_activity(
            call_llm_activity, name="call_llm_activity"
        )
        self._workflow_runtime.register_activity(
            execute_tool_activity, name="execute_tool_activity"
        )

        # Register agent's tools in the global registry
        self._register_agent_tools()

        # Create workflow client (for starting/managing workflows)
        self._workflow_client: Optional[DaprWorkflowClient] = None
        self._started = False

    def _register_agent_tools(self) -> None:
        """Register the agent's tools in the global tool registry."""
        clear_tool_registry()

        # Get tools from agent
        tools = getattr(self._agent, "tools", []) or []

        for tool in tools:
            register_tool(tool.name, tool)
            logger.info(f"Registered tool: {tool.name}")

    def _get_agent_config(self) -> AgentConfig:
        """Extract serializable agent configuration."""
        # Get tools
        tools = getattr(self._agent, "tools", []) or []
        tool_definitions = []

        for tool in tools:
            # Try to get tool declaration/schema
            declaration = None
            if hasattr(tool, "_get_declaration"):
                try:
                    declaration = tool._get_declaration()
                except Exception:
                    declaration = None

            parameters = None
            if declaration and hasattr(declaration, "parameters"):
                # Convert parameters schema to dict safely
                params = declaration.parameters
                if params:
                    try:
                        # Try model_dump for pydantic v2
                        if hasattr(params, "model_dump"):
                            parameters = params.model_dump(exclude_none=True)
                        # Try to_dict for other objects
                        elif hasattr(params, "to_dict"):
                            parameters = params.to_dict()
                        # Try JSON serialization as fallback
                        else:
                            import json

                            parameters = json.loads(json.dumps(params, default=str))
                    except Exception:
                        # Skip parameters if we can't serialize them
                        parameters = None

            tool_definitions.append(
                ToolDefinition(
                    name=tool.name,
                    description=getattr(tool, "description", "") or "",
                    parameters=parameters,
                )
            )

        # Get system instruction
        system_instruction: Optional[str] = None
        if hasattr(self._agent, "instruction"):
            instr = self._agent.instruction
            if isinstance(instr, str):
                system_instruction = instr
        elif hasattr(self._agent, "system_instruction"):
            instr = self._agent.system_instruction
            if isinstance(instr, str):
                system_instruction = instr

        # Get model name as string
        model = getattr(self._agent, "model", "gemini-2.0-flash")
        if not isinstance(model, str):
            model = str(model)

        return AgentConfig(
            name=self._agent.name,
            model=model,
            system_instruction=system_instruction,
            tool_definitions=tool_definitions,
        )

    def start(self) -> None:
        """Start the workflow runtime.

        This must be called before running any workflows. It starts listening
        for workflow work items in the background.
        """
        if self._started:
            return

        self._workflow_runtime.start()
        self._workflow_client = DaprWorkflowClient(host=self._host, port=self._port)
        self._started = True
        logger.info("Dapr Workflow runtime started")

    def shutdown(self) -> None:
        """Shutdown the workflow runtime.

        Call this when you're done running workflows to clean up resources.
        """
        if not self._started:
            return

        self._workflow_runtime.shutdown()
        self._started = False
        logger.info("Dapr Workflow runtime stopped")

    async def run_async(
        self,
        user_message: str,
        session_id: str,
        *,
        user_id: Optional[str] = None,
        app_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.5,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the agent with a user message.

        This starts a new Dapr Workflow for the agent execution. Each tool
        execution becomes a durable activity within the workflow.

        Args:
            user_message: The user's input message
            session_id: Session ID for the conversation
            user_id: Optional user ID
            app_name: Optional application name
            workflow_id: Optional workflow instance ID (generated if not provided)
            poll_interval: How often to poll for workflow status (seconds)

        Yields:
            Event dictionaries with workflow progress updates

        Raises:
            RuntimeError: If the runner hasn't been started
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")

        assert self._workflow_client is not None

        # Generate workflow ID if not provided
        if workflow_id is None:
            workflow_id = f"agent-{session_id}-{uuid.uuid4().hex[:8]}"

        # Create initial messages
        messages = [Message(role=MessageRole.USER, content=user_message)]

        # Create workflow input
        workflow_input = AgentWorkflowInput(
            agent_config=self._get_agent_config(),
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            app_name=app_name,
            iteration=0,
            max_iterations=self._max_iterations,
        )

        # Convert to dict and verify JSON serializable
        import json

        workflow_input_dict = workflow_input.to_dict()
        # Test serialization - this will raise if not serializable
        json.dumps(workflow_input_dict)

        # Start the workflow
        logger.info(f"Starting workflow: {workflow_id}")
        self._workflow_client.schedule_new_workflow(
            workflow=adk_agent_workflow,
            input=workflow_input_dict,
            instance_id=workflow_id,
        )

        # Yield start event
        yield {
            "type": "workflow_started",
            "workflow_id": workflow_id,
            "session_id": session_id,
        }

        # Poll for workflow completion
        import asyncio

        previous_status = None

        while True:
            await asyncio.sleep(poll_interval)

            state = self._workflow_client.get_workflow_state(instance_id=workflow_id)

            if state is None:
                yield {
                    "type": "workflow_error",
                    "workflow_id": workflow_id,
                    "error": "Workflow state not found",
                }
                break

            # Yield status change events
            if state.runtime_status != previous_status:
                yield {
                    "type": "workflow_status_changed",
                    "workflow_id": workflow_id,
                    "status": str(state.runtime_status),
                    "custom_status": state.serialized_custom_status,
                }
                previous_status = state.runtime_status

            # Check for completion
            if state.runtime_status == WorkflowStatus.COMPLETED:
                # Parse output
                output_data = state.serialized_output
                if output_data:
                    try:
                        import json

                        output_dict = (
                            json.loads(output_data)
                            if isinstance(output_data, str)
                            else output_data
                        )
                        output = AgentWorkflowOutput.from_dict(output_dict)

                        yield {
                            "type": "workflow_completed",
                            "workflow_id": workflow_id,
                            "final_response": output.final_response,
                            "iterations": output.iterations,
                            "status": output.status,
                        }
                    except Exception as e:
                        yield {
                            "type": "workflow_completed",
                            "workflow_id": workflow_id,
                            "raw_output": output_data,
                            "parse_error": str(e),
                        }
                else:
                    yield {
                        "type": "workflow_completed",
                        "workflow_id": workflow_id,
                    }
                break

            elif state.runtime_status == WorkflowStatus.FAILED:
                # Extract failure details
                error_info = None
                if state.failure_details:
                    fd = state.failure_details
                    error_info = {
                        "message": getattr(fd, "message", str(fd)),
                        "error_type": getattr(fd, "error_type", None),
                        "stack_trace": getattr(fd, "stack_trace", None),
                    }
                yield {
                    "type": "workflow_failed",
                    "workflow_id": workflow_id,
                    "error": error_info,
                }
                break

            elif state.runtime_status == WorkflowStatus.TERMINATED:
                yield {
                    "type": "workflow_terminated",
                    "workflow_id": workflow_id,
                }
                break

    def get_workflow_status(self, workflow_id: str) -> Optional[dict[str, Any]]:
        """Get the status of a workflow.

        Args:
            workflow_id: The workflow instance ID

        Returns:
            Dictionary with workflow status or None if not found
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")

        assert self._workflow_client is not None

        state = self._workflow_client.get_workflow_state(instance_id=workflow_id)
        if state is None:
            return None

        return {
            "workflow_id": workflow_id,
            "status": str(state.runtime_status),
            "custom_status": state.serialized_custom_status,
            "created_at": str(state.created_at) if state.created_at else None,
            "last_updated_at": str(state.last_updated_at)
            if state.last_updated_at
            else None,
        }

    def terminate_workflow(self, workflow_id: str) -> None:
        """Terminate a running workflow.

        Args:
            workflow_id: The workflow instance ID
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")

        assert self._workflow_client is not None

        self._workflow_client.terminate_workflow(instance_id=workflow_id)
        logger.info(f"Terminated workflow: {workflow_id}")

    def purge_workflow(self, workflow_id: str) -> None:
        """Purge a completed or terminated workflow.

        This removes all workflow state from the state store.

        Args:
            workflow_id: The workflow instance ID
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")

        assert self._workflow_client is not None

        self._workflow_client.purge_workflow(instance_id=workflow_id)
        logger.info(f"Purged workflow: {workflow_id}")

    @property
    def agent(self) -> "LlmAgent":
        """The ADK agent being executed."""
        return self._agent

    @property
    def is_running(self) -> bool:
        """Whether the workflow runtime is running."""
        return self._started
