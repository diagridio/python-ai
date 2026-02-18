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

"""Runner for executing OpenAI Agents SDK agents as Dapr Workflows."""

import json
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
    openai_agents_workflow,
    call_llm_activity,
    execute_tool_activity,
    register_tool,
    clear_tool_registry,
)

if TYPE_CHECKING:
    from agents import Agent

logger = logging.getLogger(__name__)


class DaprWorkflowAgentRunner:
    """Runner that executes OpenAI Agents SDK agents as Dapr Workflows.

    This runner wraps an OpenAI Agents SDK Agent and executes it using Dapr
    Workflows, making each tool execution a durable activity. This provides:

    - Fault tolerance: Agents automatically resume from the last successful activity
    - Durability: Agent state persists and survives process restarts
    - Observability: Full visibility into agent execution through Dapr's workflow APIs

    Example:
        ```python
        from agents import Agent, function_tool
        from diagrid.agent.openai_agents import DaprWorkflowAgentRunner

        @function_tool
        def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        agent = Agent(
            name="weather_agent",
            instructions="You help users check weather.",
            model="gpt-4o-mini",
            tools=[get_weather],
        )

        runner = DaprWorkflowAgentRunner(agent=agent)
        runner.start()

        async for event in runner.run_async(
            user_message="What's the weather in Tokyo?",
            session_id="session-123",
        ):
            print(event)

        runner.shutdown()
        ```

    Attributes:
        agent: The OpenAI Agents SDK Agent to execute
        workflow_runtime: The Dapr WorkflowRuntime instance
        workflow_client: The Dapr WorkflowClient for managing workflows
    """

    def __init__(
        self,
        agent: "Agent",
        *,
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_iterations: int = 25,
    ):
        """Initialize the runner.

        Args:
            agent: The OpenAI Agents SDK Agent to execute
            host: Dapr sidecar host (default: localhost)
            port: Dapr sidecar port (default: 50001)
            max_iterations: Maximum number of LLM call iterations (default: 25)
        """
        self._agent = agent
        self._max_iterations = max_iterations
        self._host = host
        self._port = port

        # Create workflow runtime
        self._workflow_runtime = WorkflowRuntime(host=host, port=port)

        # Register workflow and activities
        self._workflow_runtime.register_workflow(
            openai_agents_workflow, name="openai_agents_agent_workflow"
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
            tool_name = getattr(tool, "name", None)
            if not tool_name:
                # Fallback to function name
                if hasattr(tool, "_fn"):
                    tool_name = tool._fn.__name__
                elif hasattr(tool, "__name__"):
                    tool_name = tool.__name__
                else:
                    tool_name = str(type(tool).__name__)

            tool_def = self._create_tool_definition(tool, tool_name)
            register_tool(tool_name, tool, tool_def)
            logger.info(f"Registered tool: {tool_name}")

    def _create_tool_definition(self, tool: Any, name: str) -> ToolDefinition:
        """Create a serializable tool definition from an OpenAI Agents SDK tool."""
        # Get description
        description = getattr(tool, "description", "") or ""

        # Try to extract parameters schema via params_json_schema
        parameters = None
        if hasattr(tool, "params_json_schema"):
            try:
                schema = tool.params_json_schema
                if callable(schema):
                    parameters = schema()
                else:
                    parameters = schema
            except Exception:
                pass

        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        )

    def _get_agent_config(self) -> AgentConfig:
        """Extract serializable agent configuration."""
        # Get tools
        tools = getattr(self._agent, "tools", []) or []
        tool_definitions = []

        for tool in tools:
            tool_name = getattr(tool, "name", None)
            if not tool_name:
                if hasattr(tool, "_fn"):
                    tool_name = tool._fn.__name__
                elif hasattr(tool, "__name__"):
                    tool_name = tool.__name__
                else:
                    tool_name = str(type(tool).__name__)
            tool_definitions.append(self._create_tool_definition(tool, tool_name))

        # Get model name
        model = getattr(self._agent, "model", "gpt-4o-mini")
        if not isinstance(model, str):
            model = str(model)

        # Get instructions
        instructions = getattr(self._agent, "instructions", "") or ""

        return AgentConfig(
            name=self._agent.name,
            instructions=instructions,
            model=model,
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
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.5,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the agent with a user message.

        This starts a new Dapr Workflow for the agent execution. Each tool
        execution becomes a durable activity within the workflow.

        Args:
            user_message: The user's input message
            session_id: Session ID for the conversation
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
            workflow_id = f"openai-agents-{session_id}-{uuid.uuid4().hex[:8]}"

        # Create initial messages
        messages = [Message(role=MessageRole.USER, content=user_message)]

        # Create workflow input
        workflow_input = AgentWorkflowInput(
            agent_config=self._get_agent_config(),
            messages=messages,
            session_id=session_id,
            iteration=0,
            max_iterations=self._max_iterations,
        )

        # Convert to dict and verify JSON serializable
        workflow_input_dict = workflow_input.to_dict()
        json.dumps(workflow_input_dict)  # Validate serialization

        # Start the workflow
        logger.info(f"Starting workflow: {workflow_id}")
        self._workflow_client.schedule_new_workflow(
            workflow=openai_agents_workflow,
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
                output_data = state.serialized_output
                if output_data:
                    try:
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

    def run_sync(
        self,
        user_message: str,
        session_id: str,
        *,
        workflow_id: Optional[str] = None,
        timeout: float = 300.0,
    ) -> AgentWorkflowOutput:
        """Run the agent synchronously and wait for completion.

        Args:
            user_message: The user's input message
            session_id: Session ID for the conversation
            workflow_id: Optional workflow instance ID (generated if not provided)
            timeout: Maximum time to wait for completion (seconds)

        Returns:
            AgentWorkflowOutput with the final result

        Raises:
            RuntimeError: If the runner hasn't been started
            TimeoutError: If the workflow doesn't complete in time
        """
        import asyncio

        async def _run():
            result = None
            async for event in self.run_async(
                user_message=user_message,
                session_id=session_id,
                workflow_id=workflow_id,
            ):
                if event["type"] == "workflow_completed":
                    result = AgentWorkflowOutput(
                        final_response=event.get("final_response"),
                        messages=[],  # Not included in event
                        iterations=event.get("iterations", 0),
                        status=event.get("status", "completed"),
                    )
                elif event["type"] == "workflow_failed":
                    error = event.get("error", {})
                    raise RuntimeError(
                        f"Workflow failed: {error.get('message', 'Unknown error')}"
                    )
                elif event["type"] == "workflow_error":
                    raise RuntimeError(f"Workflow error: {event.get('error')}")
            return result

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(asyncio.wait_for(_run(), timeout=timeout))
        finally:
            loop.close()

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

    def serve(self, *, port: int = 5001, host: str = "0.0.0.0") -> None:
        """Start an HTTP server exposing /agent/run endpoints.

        Requires: pip install fastapi uvicorn

        Args:
            port: Port to listen on (default: 5001)
            host: Host to bind to (default: 0.0.0.0)
        """
        try:
            from fastapi import FastAPI, HTTPException
            import uvicorn
        except ImportError:
            raise ImportError(
                "fastapi and uvicorn are required for serve(). "
                "Install them with: pip install fastapi uvicorn[standard]"
            )

        app = FastAPI()
        self.start()

        @app.post("/agent/run")
        async def run_agent(request: dict) -> dict:  # type: ignore[type-arg]
            session_id = request.get("session_id", uuid.uuid4().hex[:8])
            task = request.get("task", "")
            result: dict[str, Any] = {}
            async for event in self.run_async(user_message=task, session_id=session_id):
                if event["type"] == "workflow_started":
                    result["instance_id"] = event["workflow_id"]
                elif event["type"] == "workflow_completed":
                    result.update(event)
                    break
                elif event["type"] == "workflow_failed":
                    result.update(event)
                    break
            return result

        @app.get("/agent/run/{workflow_id}")
        async def get_status(workflow_id: str) -> dict:  # type: ignore[type-arg]
            status = self.get_workflow_status(workflow_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Workflow not found")
            return status

        uvicorn.run(app, host=host, port=port)

    @property
    def agent(self) -> "Agent":
        """The OpenAI Agents SDK agent being executed."""
        return self._agent

    @property
    def is_running(self) -> bool:
        """Whether the workflow runtime is running."""
        return self._started
