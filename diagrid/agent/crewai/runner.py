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

"""Runner for executing CrewAI agents as Dapr Workflows."""

import json
import logging
import uuid
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING

from diagrid.agent.core.chat import DaprChatClient
from diagrid.agent.core.workflow import BaseWorkflowRunner

from .models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    Message,
    MessageRole,
    TaskConfig,
    ToolDefinition,
)
from .workflow import (
    agent_workflow,
    call_llm_activity,
    execute_tool_activity,
    register_tool,
    clear_tool_registry,
    set_default_workflow_input_factory,
)

if TYPE_CHECKING:
    from crewai import Agent, Task

logger = logging.getLogger(__name__)


class DaprWorkflowAgentRunner(BaseWorkflowRunner):
    """Runner that executes CrewAI agents as Dapr Workflows.

    This runner wraps a CrewAI Agent and executes it using Dapr Workflows,
    making each tool execution a durable activity. This provides:

    - Fault tolerance: Agents automatically resume from the last successful activity
    - Durability: Agent state persists and survives process restarts
    - Observability: Full visibility into agent execution through Dapr's workflow APIs

    Example:
        ```python
        from crewai import Agent, Task
        from crewai.tools import tool
        from diagrid.agent.crewai import DaprWorkflowAgentRunner

        @tool("Search the web")
        def search_web(query: str) -> str:
            return f"Results for: {query}"

        # Create your CrewAI agent
        agent = Agent(
            role="Research Assistant",
            goal="Help users find information",
            backstory="An expert researcher",
            tools=[search_web],
            llm="openai/gpt-4o-mini",
        )

        # Create runner and start the workflow runtime
        runner = DaprWorkflowAgentRunner(agent=agent)
        runner.start()

        # Run the agent - each tool call is now durable
        async for event in runner.run_async(task=task, session_id="session-123"):
            print(event)

        # Shutdown when done
        runner.shutdown()
        ```

    Attributes:
        agent: The CrewAI Agent to execute
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
        component_name: Optional[str] = None,
        state_store: Optional[Any] = None,
    ):
        """Initialize the runner.

        Args:
            agent: The CrewAI Agent to execute
            host: Dapr sidecar host (default: localhost)
            port: Dapr sidecar port (default: 50001)
            max_iterations: Maximum number of LLM call iterations
                           (default: uses agent's max_iter)
            registry_config: Optional registry configuration for metadata extraction
            component_name: Dapr conversation component name. If provided, always
                uses Dapr Conversation API. If None and agent has no LLM configured,
                auto-detects a conversation component.
        """
        self._agent = agent

        _max_iter = max_iterations or int(getattr(agent, "max_iter", 25))

        super().__init__(
            host=host,
            port=port,
            max_iterations=_max_iter,
            component_name=component_name,
            state_store=state_store,
        )

        # Auto-detect: if user provided no LLM on the agent, resolve a component
        if self._component_name is None and not self._agent_has_llm():
            self._dapr_chat_client = DaprChatClient()
            self._component_name = self._dapr_chat_client.component_name
            logger.info(
                "No LLM configured on agent; using Dapr conversation component: %s",
                self._component_name,
            )
        elif self._component_name is not None:
            self._dapr_chat_client = DaprChatClient(component_name=self._component_name)

        # Register metadata
        self._register_agent_metadata(
            agent=self._agent,
            framework="crewai",
            registry=registry_config,
            component_name=self._component_name,
            state_store_name=self._state_store.store_name
            if self._state_store
            else None,
        )

        # Register workflow and activities
        self._register_workflow_components()

        # Register agent's tools in the global registry
        self._register_agent_tools()

    def _agent_has_llm(self) -> bool:
        """Check if the agent has an explicit LLM configured."""
        llm = getattr(self._agent, "llm", None)
        if llm is None:
            return False
        # CrewAI uses _NotSpecified sentinel for unset fields
        if type(llm).__name__ == "_NotSpecified":
            return False
        return True

    def _register_workflow_components(self) -> None:
        """Register workflow and activities on the workflow runtime."""
        self._workflow_runtime.register_workflow(agent_workflow, name="agent_workflow")
        self._workflow_runtime.register_activity(
            call_llm_activity, name="call_llm_activity"
        )
        self._workflow_runtime.register_activity(
            execute_tool_activity, name="execute_tool_activity"
        )

    def _register_agent_tools(self) -> None:
        """Register the agent's tools in the global tool registry."""
        clear_tool_registry()

        # Get tools from agent
        tools = getattr(self._agent, "tools", []) or []

        for tool in tools:
            tool_name = self._get_tool_name(tool)
            tool_def = self._create_tool_definition(tool)
            register_tool(tool_name, tool, tool_def)
            logger.info(f"Registered tool: {tool_name}")

    def _get_tool_name(self, tool: Any) -> str:
        """Get the name of a tool, sanitized for OpenAI API compatibility.

        OpenAI requires tool names to match pattern: ^[a-zA-Z0-9_-]+$
        CrewAI's @tool decorator sets `name` to the description string,
        so we prefer the underlying function name when available.
        """
        import re

        name = None

        # Try to get the underlying function name first (for @tool decorated functions)
        if hasattr(tool, "func") and hasattr(tool.func, "__name__"):
            name = tool.func.__name__
        elif hasattr(tool, "_run") and hasattr(tool._run, "__name__"):
            # For BaseTool subclasses, try to get a meaningful name
            name = getattr(tool, "name", None)
        elif hasattr(tool, "name"):
            name = tool.name
        elif hasattr(tool, "__name__"):
            name = tool.__name__
        else:
            name = str(type(tool).__name__)

        # Sanitize the name to match OpenAI's pattern: ^[a-zA-Z0-9_-]+$
        if name:
            # Replace spaces and invalid chars with underscores
            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
            # Remove consecutive underscores
            sanitized = re.sub(r"_+", "_", sanitized)
            # Remove leading/trailing underscores
            sanitized = sanitized.strip("_")
            if sanitized:
                return sanitized

        return "unknown_tool"

    def _create_tool_definition(self, tool: Any) -> ToolDefinition:
        """Create a serializable tool definition from a CrewAI tool."""
        name = self._get_tool_name(tool)

        # Get description - CrewAI @tool decorator stores it in 'description' attribute
        # but also the 'name' attribute might contain the description string
        description = getattr(tool, "description", "") or ""
        if not description and hasattr(tool, "name"):
            # If name looks like a description (has spaces), use it as description
            tool_name = getattr(tool, "name", "")
            if " " in tool_name:
                description = tool_name

        # Also try to get docstring from underlying function
        if not description:
            if hasattr(tool, "func") and tool.func.__doc__:
                description = tool.func.__doc__
            elif hasattr(tool, "_run") and tool._run.__doc__:
                description = tool._run.__doc__

        result_as_answer = getattr(tool, "result_as_answer", False)

        # Try to extract parameters schema
        parameters = None
        if hasattr(tool, "args_schema"):
            schema = tool.args_schema
            if hasattr(schema, "model_json_schema"):
                try:
                    parameters = schema.model_json_schema()
                except Exception:
                    pass
            elif hasattr(schema, "schema"):
                try:
                    parameters = schema.schema()
                except Exception:
                    pass

        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            result_as_answer=result_as_answer,
        )

    def _get_agent_config(self) -> AgentConfig:
        """Extract serializable agent configuration."""
        # Get tools
        tools = getattr(self._agent, "tools", []) or []
        tool_definitions = []

        for tool in tools:
            tool_definitions.append(self._create_tool_definition(tool))

        # Get model name
        model = "gpt-4o-mini"  # Default
        if hasattr(self._agent, "llm"):
            llm = self._agent.llm
            if isinstance(llm, str):
                model = llm
            elif hasattr(llm, "model_name"):
                model = llm.model_name
            elif hasattr(llm, "model"):
                model = str(llm.model)

        return AgentConfig(
            role=self._safe_str(self._agent.role) or "",
            goal=self._safe_str(self._agent.goal) or "",
            backstory=self._safe_str(self._agent.backstory) or "",
            model=model,
            tool_definitions=tool_definitions,
            max_iter=self._safe_int(getattr(self._agent, "max_iter", 25), 25),
            verbose=self._safe_bool(getattr(self._agent, "verbose", False)),
            allow_delegation=self._safe_bool(
                getattr(self._agent, "allow_delegation", False)
            ),
            system_template=self._safe_str(
                getattr(self._agent, "system_template", None)
            ),
            prompt_template=self._safe_str(
                getattr(self._agent, "prompt_template", None)
            ),
            response_template=self._safe_str(
                getattr(self._agent, "response_template", None)
            ),
            component_name=self._component_name,
        )

    def _get_task_config(self, task: "Task") -> TaskConfig:
        """Extract serializable task configuration."""
        return TaskConfig(
            description=self._safe_str(task.description) or "",
            expected_output=self._safe_str(task.expected_output) or "",
            context=self._safe_str(getattr(task, "context", None)),
        )

    def _safe_str(self, value: Any) -> Optional[str]:
        """Safely convert a value to string, handling CrewAI's _NotSpecified sentinel."""
        if value is None:
            return None
        if type(value).__name__ == "_NotSpecified":
            return None
        if isinstance(value, str):
            return value
        return str(value)

    def _safe_int(self, value: Any, default: int) -> int:
        """Safely convert a value to int, handling CrewAI's _NotSpecified sentinel."""
        if value is None or type(value).__name__ == "_NotSpecified":
            return default
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _safe_bool(self, value: Any) -> bool:
        """Safely convert a value to bool, handling CrewAI's _NotSpecified sentinel."""
        if value is None or type(value).__name__ == "_NotSpecified":
            return False
        return bool(value)

    # ------------------------------------------------------------------
    # Framework-specific run methods
    # ------------------------------------------------------------------

    async def run_async(
        self,
        task: "Task",
        session_id: str,
        *,
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.5,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the agent with a task.

        Args:
            task: The CrewAI Task to execute
            session_id: Session ID for the execution
            workflow_id: Optional workflow instance ID (generated if not provided)
            poll_interval: How often to poll for workflow status (seconds)

        Yields:
            Event dictionaries with workflow progress updates
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")

        assert self._workflow_client is not None

        if workflow_id is None:
            workflow_id = f"crewai-{session_id}-{uuid.uuid4().hex[:8]}"

        messages = [
            Message(
                role=MessageRole.USER,
                content=f"Please complete the following task:\n\n{task.description}\n\nExpected output: {task.expected_output}",
            )
        ]

        workflow_input = AgentWorkflowInput(
            agent_config=self._get_agent_config(),
            task_config=self._get_task_config(task),
            messages=messages,
            session_id=session_id,
            iteration=0,
            max_iterations=self._max_iterations,
        )

        workflow_input_dict = workflow_input.to_dict()
        json.dumps(workflow_input_dict)  # Validate serialization

        logger.info(f"Starting workflow: {workflow_id}")
        self._workflow_client.schedule_new_workflow(
            workflow=agent_workflow,
            input=workflow_input_dict,
            instance_id=workflow_id,
        )

        yield {
            "type": "workflow_started",
            "workflow_id": workflow_id,
            "session_id": session_id,
            "agent_role": self._agent.role,
        }

        def _parse_output(wf_id: str, output_dict: dict) -> dict:  # type: ignore[type-arg]
            output = AgentWorkflowOutput.from_dict(output_dict)
            return {
                "type": "workflow_completed",
                "workflow_id": wf_id,
                "final_response": output.final_response,
                "iterations": output.iterations,
                "status": output.status,
            }

        async for event in self._poll_workflow(
            workflow_id,
            session_id,
            poll_interval=poll_interval,
            parse_output=_parse_output,
        ):
            yield event

    def run_sync(
        self,
        task: "Task",
        session_id: str,
        *,
        workflow_id: Optional[str] = None,
        timeout: float = 300.0,
    ) -> AgentWorkflowOutput:
        """Run the agent synchronously and wait for completion.

        Args:
            task: The CrewAI Task to execute
            session_id: Session ID for the execution
            workflow_id: Optional workflow instance ID (generated if not provided)
            timeout: Maximum time to wait for completion (seconds)

        Returns:
            AgentWorkflowOutput with the final result
        """

        async def _run():
            result = None
            async for event in self.run_async(
                task=task,
                session_id=session_id,
                workflow_id=workflow_id,
            ):
                if event["type"] == "workflow_completed":
                    result = AgentWorkflowOutput(
                        final_response=event.get("final_response"),
                        messages=[],
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

        return self._run_sync(_run(), timeout=timeout)

    # ------------------------------------------------------------------
    # Serve overrides
    # ------------------------------------------------------------------

    def _setup_telemetry(self) -> None:
        from diagrid.agent.core.telemetry import patch_crewai_telemetry, instrument_grpc

        patch_crewai_telemetry()
        instrument_grpc()

    def _setup_serve_defaults(self) -> None:
        agent_config = self._get_agent_config()

        def _build_workflow_input(task_str: str) -> dict[str, Any]:
            task_config = TaskConfig(
                description=task_str,
                expected_output="A helpful response",
            )
            return AgentWorkflowInput(
                agent_config=agent_config,
                task_config=task_config,
                messages=[
                    Message(
                        role=MessageRole.USER,
                        content=(
                            f"Please complete the following task:\n\n{task_str}\n\n"
                            f"Expected output: A helpful response"
                        ),
                    )
                ],
                session_id=uuid.uuid4().hex[:8],
                iteration=0,
                max_iterations=self._max_iterations,
            ).to_dict()

        set_default_workflow_input_factory(_build_workflow_input)

    async def _serve_run(
        self,
        request: dict,
        session_id: str,  # type: ignore[type-arg]
    ) -> AsyncIterator[dict[str, Any]]:
        from crewai import Task

        task_description = request.get("task") or ""
        task = Task(
            description=task_description,
            expected_output="A helpful response",
            agent=self._agent,
        )
        async for event in self.run_async(task=task, session_id=session_id):
            yield event

    @property
    def agent(self) -> "Agent":
        """The CrewAI agent being executed."""
        return self._agent
