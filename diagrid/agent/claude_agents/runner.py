# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Runner for executing Claude Agent SDK agents as Dapr Workflows."""

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
    clear_tool_registry,
    execute_tool_activity,
    register_tool,
    set_default_workflow_input_factory,
)

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions

logger = logging.getLogger(__name__)


class DaprWorkflowAgentRunner(BaseWorkflowRunner):
    """Runner that executes a Claude Agent SDK agent as a Dapr Workflow.

    Each agent invocation runs as a single Dapr workflow instance. Within that
    workflow:

    - Each LLM turn (one Anthropic ``messages.create`` call) is its own
      ``call_llm_activity`` — durable, retried, and replayable.
    - Each tool invocation is its own ``execute_tool_activity`` — fanned out in
      parallel when the model emits multiple tool_use blocks per turn.

    This gives every step of the agent loop an independent durable boundary,
    so a crash mid-tool-call resumes exactly where it left off on restart.

    Example:
        ```python
        from claude_agent_sdk import ClaudeAgentOptions, tool
        from diagrid.agent.claude_agents import DaprWorkflowAgentRunner

        @tool("get_weather", "Get the weather for a city", {"city": str})
        async def get_weather(args):
            return {"content": [{"type": "text", "text": f"Sunny in {args['city']}"}]}

        options = ClaudeAgentOptions(
            system_prompt="You are a helpful weather assistant.",
            model="claude-sonnet-4-6",
        )

        runner = DaprWorkflowAgentRunner(
            name="weather-agent",
            options=options,
            tools=[get_weather],
        )
        runner.start()

        async for event in runner.run_async(
            user_message="What's the weather in Tokyo?",
            session_id="session-123",
        ):
            print(event)

        runner.shutdown()
        ```
    """

    def __init__(
        self,
        *,
        name: str,
        options: Optional["ClaudeAgentOptions"] = None,
        tools: Optional[list[Any]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_iterations: int = 25,
        registry_config: Optional[Any] = None,
        state_store: Optional[Any] = None,
    ) -> None:
        """Initialize the runner.

        Args:
            name: Required workflow name (sanitized into the workflow ID).
            options: Optional ``ClaudeAgentOptions``. ``system_prompt`` and
                ``model`` from this object are used when not overridden by the
                explicit arguments below.
            tools: List of tools — each can be a plain callable, a
                ``SdkMcpTool`` from ``claude_agent_sdk.tool``, or any object
                with ``name``, ``description``, ``input_schema``, and
                ``handler`` attributes.
            model: Claude model to use. Falls back to ``options.model`` then
                ``"claude-sonnet-4-6"``.
            system_prompt: System prompt. Falls back to ``options.system_prompt``
                when that is a plain string.
            max_tokens: Max output tokens per LLM call (default: 4096).
            host: Dapr sidecar host (default: localhost).
            port: Dapr sidecar port (default: 50001).
            max_iterations: Maximum LLM iterations per workflow (default: 25).
            registry_config: Optional registry configuration for metadata.
            state_store: Optional ``DaprStateStore`` for memory persistence.
        """
        self._options = options
        self._tools: list[Any] = list(tools or [])

        resolved_model = (
            model or self._resolve_option_model(options) or ("claude-sonnet-4-6")
        )
        resolved_prompt = (
            system_prompt
            if system_prompt is not None
            else self._resolve_option_system_prompt(options)
        )

        self._model = resolved_model
        self._system_prompt = resolved_prompt or ""
        self._max_tokens = max_tokens

        super().__init__(
            name,
            framework=SupportedFrameworks.CLAUDE_AGENTS,
            host=host,
            port=port,
            max_iterations=max_iterations,
            state_store=state_store,
        )

        self._register_agent_metadata(
            agent=self,
            framework=SupportedFrameworks.CLAUDE_AGENTS,
            registry=registry_config,
            state_store_name=self._state_store.store_name
            if self._state_store
            else None,
            name=self._name,
        )

        self._register_workflow_components()
        self._register_agent_tools()

    # ------------------------------------------------------------------
    # Option resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_option_model(options: Optional["ClaudeAgentOptions"]) -> Optional[str]:
        if options is None:
            return None
        model = getattr(options, "model", None)
        return model if isinstance(model, str) else None

    @staticmethod
    def _resolve_option_system_prompt(
        options: Optional["ClaudeAgentOptions"],
    ) -> Optional[str]:
        if options is None:
            return None
        sp = getattr(options, "system_prompt", None)
        # ClaudeAgentOptions accepts str or a preset dict; only str applies to
        # the direct Anthropic Messages API call we use here.
        return sp if isinstance(sp, str) else None

    # ------------------------------------------------------------------
    # Workflow component registration
    # ------------------------------------------------------------------

    def _register_workflow_components(self) -> None:
        """Register the workflow and activities with the Dapr runtime."""
        self._workflow_runtime.register_workflow(
            agent_workflow, name=self.workflow_name
        )
        self._workflow_runtime.register_activity(
            call_llm_activity, name="claude_call_llm"
        )
        self._workflow_runtime.register_activity(
            execute_tool_activity, name="claude_execute_tool"
        )

    def _register_agent_tools(self) -> None:
        """Register the agent's tools in the global tool registry."""
        clear_tool_registry()

        for tool in self._tools:
            tool_name = self._extract_tool_name(tool)
            tool_def = self._create_tool_definition(tool, tool_name)
            register_tool(tool_name, tool, tool_def)
            logger.info("Registered tool: %s", tool_name)

    @staticmethod
    def _extract_tool_name(tool: Any) -> str:
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name:
            return name
        if hasattr(tool, "__name__"):
            return str(tool.__name__)
        return type(tool).__name__

    def _create_tool_definition(self, tool: Any, name: str) -> ToolDefinition:
        """Build a serializable tool definition from a registered tool object."""
        description = getattr(tool, "description", "") or ""

        parameters: Optional[dict[str, Any]] = None
        schema = getattr(tool, "input_schema", None)
        if isinstance(schema, dict):
            if "type" in schema and "properties" in schema:
                parameters = schema
            else:
                parameters = self._dict_schema_to_json_schema(schema)
        elif schema is not None:
            try:
                from claude_agent_sdk import _typeddict_to_json_schema  # type: ignore

                parameters = _typeddict_to_json_schema(schema)
            except Exception:
                parameters = None

        if parameters is None:
            parameters = {"type": "object", "properties": {}}

        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        )

    @staticmethod
    def _dict_schema_to_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Convert a ``{"field": type}`` dict to a JSON Schema object."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }
        properties: dict[str, Any] = {}
        for field_name, field_type in schema.items():
            json_type = type_map.get(field_type, "string")
            properties[field_name] = {"type": json_type}
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
        }

    def _get_agent_config(self) -> AgentConfig:
        """Build the serializable agent config sent into the workflow."""
        tool_definitions = []
        for tool in self._tools:
            tool_name = self._extract_tool_name(tool)
            tool_definitions.append(self._create_tool_definition(tool, tool_name))

        return AgentConfig(
            name=self._name,
            system_prompt=self._system_prompt,
            model=self._model,
            tool_definitions=tool_definitions,
            max_tokens=self._max_tokens,
        )

    # ------------------------------------------------------------------
    # Run methods
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
            user_message: The user's input.
            session_id: Session ID used for metadata and workflow naming.
            workflow_id: Optional workflow instance ID (generated if absent).
            poll_interval: Seconds between workflow status polls.

        Yields:
            Event dicts describing workflow lifecycle and the final result.
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")

        assert self._workflow_client is not None

        if workflow_id is None:
            workflow_id = f"claude-agents-{session_id}-{uuid.uuid4().hex[:8]}"

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

        async def _run() -> Optional[AgentWorkflowOutput]:
            result: Optional[AgentWorkflowOutput] = None
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
                    msg = (
                        error.get("message", "Unknown error")
                        if isinstance(error, dict)
                        else str(error)
                    )
                    raise RuntimeError(f"Workflow failed: {msg}")
                elif event["type"] == "workflow_error":
                    raise RuntimeError(f"Workflow error: {event.get('error')}")
            return result

        return self._run_sync(_run(), timeout=timeout)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Serve overrides
    # ------------------------------------------------------------------

    def _setup_telemetry(self) -> None:
        from diagrid.agent.core.telemetry import instrument_grpc, setup_telemetry

        setup_telemetry(self.__class__.__name__, config=self._observability_config)
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
        task = request.get("task", "") or request.get("user_message", "")
        async for event in self.run_async(user_message=task, session_id=session_id):
            yield event

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def options(self) -> Optional["ClaudeAgentOptions"]:
        """The ClaudeAgentOptions passed in (if any)."""
        return self._options

    @property
    def tools(self) -> list[Any]:
        """Registered tools."""
        return list(self._tools)
