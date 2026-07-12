# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import json
import logging
import uuid
from typing import Any, AsyncIterator, Optional, Sequence, TYPE_CHECKING

from diagrid.agent.core.types.type import SupportedFrameworks
from diagrid.agent.core.workflow import BaseWorkflowRunner
from diagrid.agent.core.workflow.naming import sanitize_agent_name

from .models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    Message,
    ToolDefinition,
)
from .workflow import (
    agent_workflow,
    call_llm_activity,
    clear_registries,
    execute_tool_activity,
    register_model,
    register_tool,
    set_default_workflow_input_factory,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class DaprWorkflowAgentRunner(BaseWorkflowRunner):
    """Runner that executes a LangChain model + tools agent loop as a Dapr Workflow.

    LangChain 1.x has no standalone native "Agent" object to wrap — its own
    ``langchain.agents.create_agent`` compiles directly to a LangGraph graph
    (already covered by this repo's ``diagrid.agent.langgraph`` integration).
    This runner instead takes ``langchain_core`` building blocks directly — a
    ``BaseChatModel`` and a list of ``BaseTool`` — and reimplements the
    tool-calling loop as a Dapr Workflow, making each chat-model call and
    each tool execution a durable activity. This provides:

    - Fault tolerance: the agent automatically resumes from the last
      successful activity on failure or restart
    - Durability: agent state persists and survives process restarts
    - Observability: full visibility into agent execution through Dapr's
      workflow APIs

    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from diagrid.agent.langchain import DaprWorkflowAgentRunner

        @tool
        def search_web(query: str) -> str:
            \"\"\"Search the web for information.\"\"\"
            return f"Results for: {query}"

        runner = DaprWorkflowAgentRunner(
            model=ChatOpenAI(model="gpt-4o-mini"),
            tools=[search_web],
            name="search-agent",
            system_prompt="You are a helpful assistant.",
        )
        runner.start()

        async for event in runner.run_async(
            task="What is the weather in Tokyo?",
            session_id="session-123",
        ):
            print(event)

        runner.shutdown()
        ```

    Attributes:
        model: The LangChain chat model being executed
        tools: The LangChain tools available to the agent
    """

    def __init__(
        self,
        model: "BaseChatModel",
        tools: Sequence["BaseTool"] = (),
        *,
        name: str,
        system_prompt: str = "",
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_iterations: int = 25,
        registry_config: Optional[Any] = None,
        state_store: Optional[Any] = None,
    ):
        """Initialize the runner.

        Args:
            model: The LangChain chat model (e.g. ``ChatOpenAI(...)``)
            tools: LangChain tools available to the model (``@tool``-decorated
                callables or ``BaseTool`` instances)
            name: Required name for the workflow
            system_prompt: Optional system prompt prepended to every run
            host: Dapr sidecar host (default: localhost)
            port: Dapr sidecar port (default: 50001)
            max_iterations: Maximum number of LLM call iterations (default: 25)
            registry_config: Optional registry configuration for metadata extraction
            state_store: Optional DaprStateStore for agent memory persistence.
        """
        self._model = model
        self._tools = list(tools)
        self._system_prompt = system_prompt
        self._sanitized_name = sanitize_agent_name(name)

        super().__init__(
            name,
            framework=SupportedFrameworks.LANGCHAIN,
            host=host,
            port=port,
            max_iterations=max_iterations,
            state_store=state_store,
        )

        # Register metadata. There is no separate native "Agent" object for
        # LangChain (see class docstring) — the runner itself carries the
        # configuration, matching the Claude Agent SDK integration.
        self._register_agent_metadata(
            agent=self,
            framework=SupportedFrameworks.LANGCHAIN,
            registry=registry_config,
            state_store_name=self._state_store.store_name
            if self._state_store
            else None,
            name=self._name,
        )

        self._register_workflow_components()
        self._register_model_and_tools()

    def _register_workflow_components(self) -> None:
        """Register workflow and activities on the workflow runtime."""
        self._workflow_runtime.register_workflow(
            agent_workflow, name=self.workflow_name
        )
        self._workflow_runtime.register_activity(
            call_llm_activity, name="langchain_call_llm_activity"
        )
        self._workflow_runtime.register_activity(
            execute_tool_activity, name="langchain_execute_tool_activity"
        )

    def _register_model_and_tools(self) -> None:
        """Register the model and tools in the global process registry."""
        clear_registries()
        register_model(self._sanitized_name, self._model)
        for tool in self._tools:
            register_tool(tool.name, tool)
            logger.info("Registered tool: %s", tool.name)

    def _get_agent_config(self) -> AgentConfig:
        """Extract serializable agent configuration."""
        from langchain_core.utils.function_calling import convert_to_openai_tool

        tool_definitions = []
        for tool in self._tools:
            schema = convert_to_openai_tool(tool)["function"]
            tool_definitions.append(
                ToolDefinition(
                    name=schema["name"],
                    description=schema.get("description", ""),
                    parameters=schema.get("parameters") or {},
                )
            )

        return AgentConfig(
            name=self._name,
            model_key=self._sanitized_name,
            system_prompt=self._system_prompt or None,
            tool_definitions=tool_definitions,
        )

    # ------------------------------------------------------------------
    # Framework-specific run methods
    # ------------------------------------------------------------------

    async def run_async(
        self,
        task: str,
        session_id: str,
        *,
        workflow_id: Optional[str] = None,
        poll_interval: float = 0.5,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the agent with a task.

        Args:
            task: The user task to send to the agent
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
            workflow_id = f"langchain-{session_id}-{uuid.uuid4().hex[:8]}"

        messages = []
        if self._system_prompt:
            messages.append(Message(role="system", content=self._system_prompt))
        messages.append(Message(role="user", content=task))

        workflow_input = AgentWorkflowInput(
            agent_config=self._get_agent_config(),
            messages=messages,
            session_id=session_id,
            iteration=0,
            max_iterations=self._max_iterations,
        )

        workflow_input_dict = workflow_input.to_dict()
        json.dumps(workflow_input_dict)

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
                "result": output.final_response,
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
        task: str,
        session_id: str,
        *,
        workflow_id: Optional[str] = None,
        timeout: float = 300.0,
    ) -> AgentWorkflowOutput:
        """Run the agent synchronously and wait for completion."""

        async def _run() -> Optional[AgentWorkflowOutput]:
            result = None
            async for event in self.run_async(
                task=task,
                session_id=session_id,
                workflow_id=workflow_id,
            ):
                if event["type"] == "workflow_completed":
                    result = AgentWorkflowOutput(
                        final_response=event.get("result"),
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
        system_prompt = self._system_prompt

        def _build_workflow_input(task_str: str) -> dict[str, Any]:
            messages = []
            if system_prompt:
                messages.append(Message(role="system", content=system_prompt))
            messages.append(Message(role="user", content=task_str))
            return AgentWorkflowInput(
                agent_config=agent_config,
                messages=messages,
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
        async for event in self.run_async(task=task, session_id=session_id):
            yield event

    @property
    def model(self) -> "BaseChatModel":
        """The LangChain chat model being executed."""
        return self._model

    @property
    def tools(self) -> list["BaseTool"]:
        """The LangChain tools available to the agent."""
        return self._tools
