# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Runner for executing OpenAI Agents SDK agents as Dapr Workflows."""

import json
import logging
import uuid
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING

from diagrid.agent.core.types.type import SupportedFrameworks
from diagrid.agent.core.workflow import BaseWorkflowRunner

from .models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    Message,
    MessageRole,
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
    from agents import Agent

logger = logging.getLogger(__name__)


class DaprWorkflowAgentRunner(BaseWorkflowRunner):
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
        name: str,
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_iterations: int = 25,
        registry_config: Optional[Any] = None,
        state_store: Optional[Any] = None,
    ):
        """Initialize the runner.

        Args:
            agent: The OpenAI Agents SDK Agent to execute
            name: Required name for the workflow
            host: Dapr sidecar host (default: localhost)
            port: Dapr sidecar port (default: 50001)
            max_iterations: Maximum number of LLM call iterations (default: 25)
            registry_config: Optional registry configuration for metadata extraction
            state_store: Optional DaprStateStore for agent memory persistence.
        """
        self._agent = agent

        super().__init__(
            name,
            framework=SupportedFrameworks.OPENAI,
            host=host,
            port=port,
            max_iterations=max_iterations,
            state_store=state_store,
        )

        # Register metadata
        self._register_agent_metadata(
            agent=self._agent,
            framework=SupportedFrameworks.OPENAI,
            registry=registry_config,
            state_store_name=self._state_store.store_name
            if self._state_store
            else None,
            name=self._name,
        )

        # Register workflow and activities
        self._register_workflow_components()

        # Register agent's tools in the global registry
        self._register_agent_tools()

    def _register_workflow_components(self) -> None:
        """Register workflow and activities on the workflow runtime."""
        self._workflow_runtime.register_workflow(
            agent_workflow, name=self.workflow_name
        )
        self._workflow_runtime.register_activity(
            call_llm_activity, name="call_llm_activity"
        )
        self._workflow_runtime.register_activity(
            execute_tool_activity, name="execute_tool_activity"
        )

    def _register_agent_tools(self) -> None:
        """Register the agent's tools in the global tool registry."""
        clear_tool_registry()

        tools = getattr(self._agent, "tools", []) or []

        for tool in tools:
            tool_name = getattr(tool, "name", None)
            if not tool_name:
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
        description = getattr(tool, "description", "") or ""

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

        model = getattr(self._agent, "model", "gpt-4o-mini")
        if not isinstance(model, str):
            model = str(model)

        instructions = getattr(self._agent, "instructions", "") or ""

        return AgentConfig(
            name=self._agent.name,
            instructions=instructions,
            model=model,
            tool_definitions=tool_definitions,
        )

    # ------------------------------------------------------------------
    # Framework-specific run methods
    # ------------------------------------------------------------------

    async def run_async(
        self,
        user_message: str,
        session_id: str,
        *,
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.5,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the agent with a user message.

        Args:
            user_message: The user's input message
            session_id: Session ID for the conversation
            workflow_id: Optional workflow instance ID (generated if not provided)
            poll_interval: How often to poll for workflow status (seconds)

        Yields:
            Event dictionaries with workflow progress updates
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")

        assert self._workflow_client is not None

        if workflow_id is None:
            workflow_id = f"openai-agents-{session_id}-{uuid.uuid4().hex[:8]}"

        messages = [Message(role=MessageRole.USER, content=user_message)]

        workflow_input = AgentWorkflowInput(
            agent_config=self._get_agent_config(),
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
        user_message: str,
        session_id: str,
        *,
        workflow_id: Optional[str] = None,
        timeout: float = 300.0,
    ) -> AgentWorkflowOutput:
        """Run the agent synchronously and wait for completion."""

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
        from diagrid.agent.core.telemetry import (
            setup_telemetry,
            instrument_grpc,
            OtelTracingProcessor,
        )

        provider = setup_telemetry(
            self.__class__.__name__, config=self._observability_config
        )
        if provider:
            try:
                from agents import add_trace_processor

                add_trace_processor(OtelTracingProcessor(provider))  # type: ignore[arg-type]
            except Exception:
                logger.debug("OpenAI Agents OTEL bridge skipped", exc_info=True)
        instrument_grpc(config=self._observability_config)

    def _setup_serve_defaults(self) -> None:
        agent_config = self._get_agent_config()

        def _build_workflow_input(task_str: str) -> dict[str, Any]:
            return AgentWorkflowInput(
                agent_config=agent_config,
                messages=[Message(role=MessageRole.USER, content=task_str)],
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
        task = request.get("task", "")
        async for event in self.run_async(user_message=task, session_id=session_id):
            yield event

    @property
    def agent(self) -> "Agent":
        """The OpenAI Agents SDK agent being executed."""
        return self._agent
