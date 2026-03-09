# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Runner for executing Pydantic AI agents as Dapr Workflows."""

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
    ToolDefinition,
)
from .utils import get_pydantic_ai_tools
from .workflow import (
    agent_workflow,
    call_llm_activity,
    execute_tool_activity,
    register_tool,
    clear_tool_registry,
    set_default_workflow_input_factory,
)

if TYPE_CHECKING:
    from pydantic_ai import Agent

logger = logging.getLogger(__name__)


class DaprWorkflowAgentRunner(BaseWorkflowRunner):
    """Runner that executes Pydantic AI agents as Dapr Workflows.

    This runner wraps a Pydantic AI Agent and executes it using Dapr
    Workflows, making each tool execution a durable activity. This provides:

    - Fault tolerance: Agents automatically resume from the last successful activity
    - Durability: Agent state persists and survives process restarts
    - Observability: Full visibility into agent execution through Dapr's workflow APIs

    Example:
        ```python
        from pydantic_ai import Agent
        from diagrid.agent.pydantic_ai import DaprWorkflowAgentRunner

        def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        agent = Agent(
            "openai:gpt-4o-mini",
            system_prompt="You help users check weather.",
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
        agent: The Pydantic AI Agent to execute
        workflow_runtime: The Dapr WorkflowRuntime instance
        workflow_client: The Dapr WorkflowClient for managing workflows
    """

    def __init__(
        self,
        agent: "Agent",  # type: ignore[type-arg]
        *,
        name: str,
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_iterations: int = 25,
        registry_config: Optional[Any] = None,
        component_name: Optional[str] = None,
        state_store: Optional[Any] = None,
    ):
        """Initialize the runner.

        Args:
            agent: The Pydantic AI Agent to execute
            name: Required name for the workflow
            host: Dapr sidecar host (default: localhost)
            port: Dapr sidecar port (default: 50001)
            max_iterations: Maximum number of LLM call iterations (default: 25)
            registry_config: Optional registry configuration for metadata extraction
            component_name: Dapr conversation component name. If provided, always
                uses Dapr Conversation API. If None and agent has no model configured,
                auto-detects a conversation component.
            state_store: Optional DaprStateStore for agent memory persistence.
        """
        self._agent = agent

        super().__init__(
            name,
            framework="pydantic_ai",
            host=host,
            port=port,
            max_iterations=max_iterations,
            component_name=component_name,
            state_store=state_store,
        )

        # Auto-detect: if user provided no model on the agent, resolve a component
        if self._component_name is None and not self._agent_has_model():
            self._dapr_chat_client = DaprChatClient()
            self._component_name = self._dapr_chat_client.component_name
            logger.info(
                "No model configured on agent; using Dapr conversation component: %s",
                self._component_name,
            )
        elif self._component_name is not None:
            self._dapr_chat_client = DaprChatClient(component_name=self._component_name)

        # Register metadata
        self._register_agent_metadata(
            agent=self._agent,
            framework="pydantic_ai",
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

    def _agent_has_model(self) -> bool:
        """Check if the agent has an explicit model configured."""
        model = getattr(self._agent, "model", None)
        if model is None:
            return False
        if isinstance(model, str) and model == "":
            return False
        return True

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

        function_tools = self._get_function_tools()

        for tool_name, tool_info in function_tools.items():
            # Extract the underlying callable from the tool info
            tool_callable = getattr(tool_info, "function", None)
            if tool_callable is None:
                tool_callable = tool_info

            tool_def = self._create_tool_definition(tool_info, tool_name)
            register_tool(tool_name, tool_callable, tool_def)
            logger.info("Registered tool: %s", tool_name)

    def _get_function_tools(self) -> dict[str, Any]:
        """Get function tools dict from the agent, supporting both old and new APIs."""
        return get_pydantic_ai_tools(self._agent)

    def _create_tool_definition(self, tool: Any, name: str) -> ToolDefinition:
        """Create a serializable tool definition from a Pydantic AI tool."""
        description = getattr(tool, "description", "") or ""

        parameters = None
        # pydantic-ai v1.61.0+: function_schema.json_schema
        func_schema = getattr(tool, "function_schema", None)
        if func_schema is not None:
            json_schema = getattr(func_schema, "json_schema", None)
            if json_schema is not None:
                parameters = json_schema
        # Fallback for older versions
        elif hasattr(tool, "parameters_json_schema"):
            try:
                schema = tool.parameters_json_schema
                if callable(schema):
                    parameters = schema()
                else:
                    parameters = schema
            except Exception:
                logger.debug("Failed to extract parameters_json_schema", exc_info=True)

        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        )

    def _get_agent_config(self) -> AgentConfig:
        """Extract serializable agent configuration."""
        function_tools = self._get_function_tools()
        tool_definitions = []

        for tool_name, tool_info in function_tools.items():
            tool_definitions.append(self._create_tool_definition(tool_info, tool_name))

        model = getattr(self._agent, "model", None)
        if model is not None and not isinstance(model, str):
            # Use model_name first (bare name), then model_id, then str()
            model = (
                getattr(model, "model_name", None)
                or getattr(model, "model_id", None)
                or str(model)
            )
        elif model is None:
            model = "unknown"

        # Extract system prompt from Pydantic AI agent
        system_prompt = ""
        system_prompts = getattr(self._agent, "_system_prompts", [])
        if system_prompts:
            # Pydantic AI stores system prompts as a tuple of (static_str | callable)
            prompt_parts = []
            for sp in system_prompts:
                if isinstance(sp, str):
                    prompt_parts.append(sp)
                elif callable(sp):
                    try:
                        result = sp()
                        if isinstance(result, str):
                            prompt_parts.append(result)
                    except Exception:
                        logger.debug(
                            "Failed to call system prompt callable", exc_info=True
                        )
            system_prompt = "\n".join(prompt_parts)

        # Fall back to _instructions (list of strings or callables)
        if not system_prompt:
            instructions = getattr(self._agent, "_instructions", [])
            if instructions:
                prompt_parts = [inst for inst in instructions if isinstance(inst, str)]
                if prompt_parts:
                    system_prompt = "\n".join(prompt_parts)

        # Fall back to system_prompt attribute if still empty
        if not system_prompt:
            system_prompt = getattr(self._agent, "system_prompt", "") or ""
            if callable(system_prompt):
                try:
                    system_prompt = system_prompt()
                except Exception:
                    logger.debug(
                        "Failed to call system_prompt attribute", exc_info=True
                    )
                    system_prompt = ""

        name = getattr(self._agent, "name", None) or "pydantic-ai-agent"

        return AgentConfig(
            name=name,
            system_prompt=system_prompt,
            model=model,
            tool_definitions=tool_definitions,
            component_name=self._component_name,
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
            workflow_id = f"pydantic-ai-{session_id}-{uuid.uuid4().hex[:8]}"

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

        logger.info("Starting workflow: %s", workflow_id)
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
        )

        setup_telemetry(self.__class__.__name__)
        instrument_grpc()

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
        request: dict[str, Any],
        session_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        task = request.get("task", "")
        async for event in self.run_async(user_message=task, session_id=session_id):
            yield event

    @property
    def agent(self) -> "Agent":  # type: ignore[type-arg]
        """The Pydantic AI agent being executed."""
        return self._agent
