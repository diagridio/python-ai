# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import json
import logging
import uuid
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING

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
    from google.adk.agents import LlmAgent

logger = logging.getLogger(__name__)


class DaprWorkflowAgentRunner(BaseWorkflowRunner):
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
        name: str,
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_iterations: int = 100,
        registry_config: Optional[Any] = None,
        state_store: Optional[Any] = None,
    ):
        """Initialize the runner.

        Args:
            agent: The ADK LlmAgent to execute
            name: Required name for the workflow
            host: Dapr sidecar host (default: localhost)
            port: Dapr sidecar port (default: 50001)
            max_iterations: Maximum number of LLM call iterations (default: 100)
            registry_config: Optional registry configuration for metadata extraction
            state_store: Optional DaprStateStore for agent memory persistence.
        """
        self._agent = agent

        super().__init__(
            name,
            framework="adk",
            host=host,
            port=port,
            max_iterations=max_iterations,
            state_store=state_store,
        )

        # Register metadata
        self._register_agent_metadata(
            agent=self._agent,
            framework="adk",
            registry=registry_config,
            state_store_name=self._state_store.store_name
            if self._state_store
            else None,
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
            register_tool(tool.name, tool)
            logger.info(f"Registered tool: {tool.name}")

    def _get_agent_config(self) -> AgentConfig:
        """Extract serializable agent configuration."""
        tools = getattr(self._agent, "tools", []) or []
        tool_definitions = []

        for tool in tools:
            declaration = None
            if hasattr(tool, "_get_declaration"):
                try:
                    declaration = tool._get_declaration()
                except Exception:
                    declaration = None

            parameters = None
            if declaration and hasattr(declaration, "parameters"):
                params = declaration.parameters
                if params:
                    try:
                        if hasattr(params, "model_dump"):
                            parameters = params.model_dump(exclude_none=True)
                        elif hasattr(params, "to_dict"):
                            parameters = params.to_dict()
                        else:
                            parameters = json.loads(json.dumps(params, default=str))
                    except Exception:
                        parameters = None

                if parameters:
                    parameters = self._normalize_schema(parameters)

            tool_definitions.append(
                ToolDefinition(
                    name=tool.name,
                    description=getattr(tool, "description", "") or "",
                    parameters=parameters,
                )
            )

        system_instruction: Optional[str] = None
        if hasattr(self._agent, "instruction"):
            instr = self._agent.instruction
            if isinstance(instr, str):
                system_instruction = instr
        elif hasattr(self._agent, "system_instruction"):
            instr = self._agent.system_instruction
            if isinstance(instr, str):
                system_instruction = instr

        model = getattr(self._agent, "model", "gemini-2.0-flash")
        if not isinstance(model, str):
            model = str(model)

        return AgentConfig(
            name=self._agent.name,
            model=model,
            system_instruction=system_instruction,
            tool_definitions=tool_definitions,
        )

    @staticmethod
    def _normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Normalize a protobuf-derived schema to standard JSON Schema.

        Google ADK's FunctionDeclaration.parameters uses Protocol Buffer
        Type enums that serialize as uppercase strings (e.g. "STRING",
        "OBJECT", "INTEGER").  The Dapr Conversation API (and OpenAI)
        expect lowercase JSON Schema types ("string", "object", "integer").

        This method recursively lowercases all "type" values.
        """
        if not isinstance(schema, dict):
            return schema

        result: dict[str, Any] = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, str):
                result[key] = value.lower()
            elif isinstance(value, dict):
                result[key] = DaprWorkflowAgentRunner._normalize_schema(value)
            elif isinstance(value, list):
                result[key] = [
                    DaprWorkflowAgentRunner._normalize_schema(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    # ------------------------------------------------------------------
    # Framework-specific run methods
    # ------------------------------------------------------------------

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

        Args:
            user_message: The user's input message
            session_id: Session ID for the conversation
            user_id: Optional user ID
            app_name: Optional application name
            workflow_id: Optional workflow instance ID (generated if not provided)
            poll_interval: How often to poll for workflow status (seconds)

        Yields:
            Event dictionaries with workflow progress updates
        """
        if not self._started:
            raise RuntimeError("Runner not started. Call start() first.")

        assert self._workflow_client is not None

        if workflow_id is None:
            workflow_id = f"agent-{session_id}-{uuid.uuid4().hex[:8]}"

        messages = [Message(role=MessageRole.USER, content=user_message)]

        workflow_input = AgentWorkflowInput(
            agent_config=self._get_agent_config(),
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            app_name=app_name,
            iteration=0,
            max_iterations=self._max_iterations,
        )

        workflow_input_dict = workflow_input.to_dict()
        json.dumps(workflow_input_dict)

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
        from diagrid.agent.core.telemetry import setup_telemetry, instrument_grpc

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
        task = request.get("task") or ""
        async for event in self.run_async(user_message=task, session_id=session_id):
            yield event

    @property
    def agent(self) -> "LlmAgent":
        """The ADK agent being executed."""
        return self._agent
