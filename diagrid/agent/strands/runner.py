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

"""Runner for executing Strands agents as Dapr Workflows."""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Generator, Optional, TYPE_CHECKING

from dapr.ext.workflow import WorkflowRuntime, DaprWorkflowClient, WorkflowStatus

from diagrid.agent.core import AgentRegistryMixin
from .workflow import WorkflowOutput

if TYPE_CHECKING:
    from strands import Agent

logger = logging.getLogger(__name__)


class DaprWorkflowAgentRunner(AgentRegistryMixin):
    """Runner that executes Strands agents as Dapr Workflows.

    This runner wraps a Strands Agent and executes it using Dapr Workflows,
    making each LLM call and tool execution a separate durable activity.
    This provides:

    - Fault tolerance: Agents automatically resume from the last successful activity
    - Durability: Agent state persists and survives process restarts
    - Observability: Full visibility into agent execution through Dapr's workflow APIs

    Architecture:
        Workflow
          +-- Activity: call_model (get next action from LLM)
          +-- Activity: execute_tool (tool call 1)
          +-- Activity: execute_tool (tool call 2)
          +-- Activity: call_model (with tool results)
          +-- ... repeat until model says done

    Example:
        ```python
        from strands import Agent, tool
        from diagrid.agent.strands import DaprWorkflowAgentRunner

        @tool
        def search_web(query: str) -> str:
            \"\"\"Search the web for information.\"\"\"
            return f"Results for: {query}"

        agent = Agent(
            model=my_model,
            tools=[search_web],
        )

        runner = DaprWorkflowAgentRunner(agent=agent)
        runner.start()

        async for event in runner.run_async(
            task="What is the weather in Tokyo?",
            session_id="session-123",
        ):
            print(event)

        runner.shutdown()
        ```

    Attributes:
        agent: The Strands Agent to execute
        workflow_runtime: The Dapr WorkflowRuntime instance
        workflow_client: The Dapr WorkflowClient for managing workflows
    """

    def __init__(
        self,
        agent: "Agent",
        *,
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_iterations: Optional[int] = None,
        registry_config: Optional[Any] = None,
    ):
        """Initialize the runner.

        Args:
            agent: The Strands Agent to execute
            host: Dapr sidecar host (default: localhost)
            port: Dapr sidecar port (default: 50001)
            max_iterations: Maximum number of LLM call iterations (default: 25)
            registry_config: Optional registry configuration for metadata extraction
        """
        self._agent = agent
        self._host = host
        self._port = port
        self._max_iterations = max_iterations or 25

        # Register metadata
        self._register_agent_metadata(
            agent=self._agent, framework="strands", registry=registry_config
        )

        # Create workflow runtime
        self._workflow_runtime = WorkflowRuntime(host=host, port=port)

        # Store reference for closures
        agent_ref = self._agent

        # ============================================================
        # Activity: Call the model (LLM)
        # ============================================================
        def call_model_activity(ctx: Any, input_data: dict) -> dict:  # type: ignore[type-arg]
            """Activity that calls the LLM model."""
            messages = input_data.get("messages", [])
            system_prompt = input_data.get("system_prompt", "")

            try:
                tool_specs = [
                    tool.tool_spec for tool in agent_ref.tool_registry.registry.values()
                ]

                async def _call() -> tuple[list[Any], str]:
                    from strands.event_loop.streaming import process_stream

                    stop_reason = "end_turn"
                    message_content: list[Any] = []

                    raw_stream = agent_ref.model.stream(
                        messages=messages,
                        tool_specs=tool_specs if tool_specs else None,
                        system_prompt=system_prompt,
                    )

                    async for event in process_stream(raw_stream):
                        if isinstance(event, dict) and "stop" in event:
                            stop_data = event["stop"]
                            if isinstance(stop_data, tuple) and len(stop_data) >= 2:
                                stop_reason = stop_data[0]
                                message = stop_data[1]
                                if isinstance(message, dict) and "content" in message:
                                    message_content = message["content"]

                    return message_content, stop_reason

                content, stop_reason = asyncio.run(_call())

                # Parse tool uses and text from the content
                tool_uses: list[Any] = []
                text_content: list[str] = []

                for block in content:
                    if isinstance(block, dict):
                        if "toolUse" in block:
                            tool_uses.append(block["toolUse"])
                        elif "text" in block:
                            text_content.append(block["text"])

                return {
                    "tool_uses": tool_uses,
                    "text": "".join(text_content),
                    "stop_reason": stop_reason,
                    "content": content,
                }

            except Exception as e:
                logger.error("call_model_activity failed: %s", str(e))
                import traceback

                traceback.print_exc()
                return {
                    "error": str(e),
                    "tool_uses": [],
                    "text": "",
                    "stop_reason": "error",
                }

        # ============================================================
        # Activity: Execute a single tool
        # ============================================================
        def execute_tool_activity(ctx: Any, input_data: dict) -> dict:  # type: ignore[type-arg]
            """Activity that executes a single tool."""
            from strands.types.tools import ToolUse

            tool_name = input_data.get("tool_name")
            tool_use_id = input_data.get("tool_use_id")
            tool_input = input_data.get("tool_input", {})

            try:
                tool = agent_ref.tool_registry.registry.get(tool_name or "")
                if not tool:
                    return {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": f"Unknown tool: {tool_name}"}],
                    }

                tool_use: ToolUse = {
                    "name": tool_name or "",
                    "toolUseId": tool_use_id or "",
                    "input": tool_input,
                }

                async def _execute() -> Any:
                    result = None
                    async for event in tool.stream(tool_use, {}):
                        result = event
                    return result

                result = asyncio.run(_execute())

                # Handle different result formats
                if isinstance(result, dict) and "toolResult" in result:
                    tool_result = result["toolResult"]
                elif isinstance(result, dict) and "status" in result:
                    tool_result = result
                else:
                    tool_result = {
                        "toolUseId": tool_use_id,
                        "status": "success",
                        "content": [{"text": str(result)}],
                    }

                return tool_result

            except Exception as e:
                logger.error("execute_tool_activity failed for %s: %s", tool_name, e)
                import traceback

                traceback.print_exc()
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": f"Error: {str(e)}"}],
                }

        # ============================================================
        # Workflow: Orchestrate model + tool calls
        # ============================================================
        max_iters = self._max_iterations

        def agent_workflow(ctx: Any, input_data: dict) -> Generator[Any, Any, dict]:  # type: ignore[type-arg]
            """Workflow that orchestrates the agent loop.

            Each iteration:
            1. Call model activity (checkpointed)
            2. If model wants tools: call each tool as a separate activity (checkpointed)
            3. Repeat until model says done
            """
            task = input_data.get("task", "")
            system_prompt = agent_ref.system_prompt or ""

            # Build initial messages
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": [{"text": task}]}
            ]
            final_text = ""
            all_tool_calls: list[dict[str, Any]] = []

            for iteration in range(max_iters):
                # Step 1: Call the model (as a durable activity)
                model_result = yield ctx.call_activity(
                    call_model_activity,
                    input={
                        "messages": messages,
                        "system_prompt": system_prompt,
                    },
                )

                if model_result.get("error"):
                    final_text = f"Error: {model_result['error']}"
                    break

                tool_uses = model_result.get("tool_uses", [])
                text = model_result.get("text", "")
                stop_reason = model_result.get("stop_reason")
                content = model_result.get("content", [])

                # Add assistant message to history
                messages.append({"role": "assistant", "content": content})

                # If no tool calls, we're done
                if not tool_uses or stop_reason == "end_turn":
                    final_text = text
                    break

                # Step 2: Execute each tool (as separate durable activities)
                tool_results = []
                for tool_use in tool_uses:
                    t_name = tool_use.get("name")
                    t_id = tool_use.get("toolUseId")
                    t_input = tool_use.get("input", {})

                    all_tool_calls.append(
                        {
                            "tool_name": t_name,
                            "tool_use_id": t_id,
                            "input": t_input,
                        }
                    )

                    # Each tool call is a separate checkpointed activity
                    result = yield ctx.call_activity(
                        execute_tool_activity,
                        input={
                            "tool_name": t_name,
                            "tool_use_id": t_id,
                            "tool_input": t_input,
                        },
                    )
                    tool_results.append({"toolResult": result})

                # Add tool results to messages
                messages.append({"role": "user", "content": tool_results})

            return {  # type: ignore[return-value]
                "result": final_text,
                "tool_calls": all_tool_calls,
            }

        # Register workflow and activities
        self._workflow_runtime.register_workflow(
            agent_workflow, name="strands_agent_workflow"
        )
        self._workflow_runtime.register_activity(
            call_model_activity, name="strands_call_model"
        )
        self._workflow_runtime.register_activity(
            execute_tool_activity, name="strands_execute_tool"
        )

        # Store references
        self._workflow_func = agent_workflow
        self._workflow_client: Optional[DaprWorkflowClient] = None
        self._started = False

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
        task: str,
        session_id: str,
        *,
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.5,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the agent with a task.

        This starts a new Dapr Workflow for the agent execution. Each LLM call
        and tool execution becomes a separate durable activity.

        Args:
            task: The user task to send to the agent
            session_id: Session ID for the execution
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
            workflow_id = f"strands-{session_id}-{uuid.uuid4().hex[:8]}"

        # Create workflow input
        input_data = {
            "task": task,
        }

        # Start the workflow
        logger.info("Starting workflow: %s", workflow_id)
        self._workflow_client.schedule_new_workflow(
            workflow=self._workflow_func,
            input=input_data,
            instance_id=workflow_id,
        )

        # Yield start event
        yield {
            "type": "workflow_started",
            "workflow_id": workflow_id,
            "session_id": session_id,
        }

        # Poll for workflow completion
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

                        yield {
                            "type": "workflow_completed",
                            "workflow_id": workflow_id,
                            "result": output_dict.get("result", ""),
                            "tool_calls": output_dict.get("tool_calls", []),
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
        task: str,
        session_id: str,
        *,
        workflow_id: Optional[str] = None,
        timeout: float = 300.0,
    ) -> WorkflowOutput:
        """Run the agent synchronously and wait for completion.

        This is a convenience method that wraps run_async and waits for
        the workflow to complete.

        Args:
            task: The user task to send to the agent
            session_id: Session ID for the execution
            workflow_id: Optional workflow instance ID (generated if not provided)
            timeout: Maximum time to wait for completion (seconds)

        Returns:
            WorkflowOutput with the agent's response

        Raises:
            RuntimeError: If the runner hasn't been started or workflow fails
            TimeoutError: If the workflow doesn't complete in time
        """

        async def _run() -> Optional[WorkflowOutput]:
            result = None
            async for event in self.run_async(
                task=task,
                session_id=session_id,
                workflow_id=workflow_id,
            ):
                if event["type"] == "workflow_completed":
                    result = WorkflowOutput(
                        result=event.get("result", ""),
                        tool_calls=event.get("tool_calls", []),
                    )
                elif event["type"] == "workflow_failed":
                    error = event.get("error", {})
                    msg = (
                        error.get("message", "Unknown error")
                        if isinstance(error, dict)
                        else str(error)
                    )
                    raise RuntimeError(f"Workflow failed: {msg}")
                elif event["type"] == "workflow_error":
                    raise RuntimeError(f"Workflow error: {event.get('error')}")
            return result

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(asyncio.wait_for(_run(), timeout=timeout))  # type: ignore[return-value]
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
        logger.info("Terminated workflow: %s", workflow_id)

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
        logger.info("Purged workflow: %s", workflow_id)

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
            task = request.get("task") or ""
            result: dict[str, Any] = {}
            async for event in self.run_async(task=task, session_id=session_id):
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
        """The Strands agent being executed."""
        return self._agent

    @property
    def is_running(self) -> bool:
        """Whether the workflow runtime is running."""
        return self._started
