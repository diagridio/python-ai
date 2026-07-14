# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import json
import logging
import uuid
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING

from diagrid.agent.core.telemetry import instrument_grpc, setup_telemetry
from diagrid.agent.core.types.type import SupportedFrameworks
from diagrid.agent.core.workflow import BaseWorkflowRunner
from diagrid.agent.core.workflow.naming import sanitize_agent_name

from .models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    ChatEntry,
    ToolDefinition,
)
from .workflow import (
    agent_workflow,
    call_model_activity,
    clear_registries,
    execute_tool_activity,
    register_model,
    register_tool,
    set_default_workflow_input_factory,
)

if TYPE_CHECKING:
    from smolagents import ToolCallingAgent  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class DaprWorkflowAgentRunner(BaseWorkflowRunner):
    """Runner that executes a smolagents ``ToolCallingAgent`` as a Dapr Workflow.

    This runner wraps a smolagents ``ToolCallingAgent`` and drives its model
    and tools as a Dapr Workflow, making each model call and each tool
    execution a separate durable activity. This provides:

    - Fault tolerance: the agent automatically resumes from the last
      successful activity on failure or restart
    - Durability: agent state persists and survives process restarts
    - Observability: full visibility into agent execution through Dapr's
      workflow APIs

    Only ``ToolCallingAgent`` is supported (not ``CodeAgent``) — ``CodeAgent``
    executes an arbitrary, model-generated Python code blob per step, which
    has no discrete per-tool-call boundary to checkpoint as a Dapr activity.
    ``ToolCallingAgent`` calls tools as structured, individually-dispatched
    JSON tool calls, so each one maps cleanly to one activity.

    Example:
        ```python
        from smolagents import ToolCallingAgent, OpenAIServerModel, tool
        from diagrid.agent.smolagents import DaprWorkflowAgentRunner

        @tool
        def search_web(query: str) -> str:
            \"\"\"Search the web for information.

            Args:
                query: The search query.
            \"\"\"
            return f"Results for: {query}"

        agent = ToolCallingAgent(
            model=OpenAIServerModel(model_id="gpt-4o-mini"),
            tools=[search_web],
        )

        runner = DaprWorkflowAgentRunner(agent=agent, name="search-agent")
        runner.start()

        async for event in runner.run_async(
            task="What is the weather in Tokyo?",
            session_id="session-123",
        ):
            print(event)

        runner.shutdown()
        ```

    Attributes:
        agent: The smolagents ToolCallingAgent being executed
    """

    def __init__(
        self,
        agent: "ToolCallingAgent",
        *,
        name: str,
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_iterations: Optional[int] = None,
        registry_config: Optional[Any] = None,
        state_store: Optional[Any] = None,
    ):
        """Initialize the runner.

        Args:
            agent: The smolagents ToolCallingAgent to execute
            name: Required name for the workflow
            host: Dapr sidecar host (default: localhost)
            port: Dapr sidecar port (default: 50001)
            max_iterations: Maximum number of model call iterations. Defaults
                to the agent's own ``max_steps``.
            registry_config: Optional registry configuration for metadata extraction
            state_store: Optional DaprStateStore for agent memory persistence.
        """
        self._agent = agent
        self._sanitized_name = sanitize_agent_name(name)
        # Captured once at construction: smolagents computes this from
        # prompt_templates + the agent's tools, and tools don't change after
        # construction, so it's safe (and much cheaper) to reuse across runs
        # rather than recomputing it inside every workflow run.
        self._system_prompt = agent.system_prompt

        super().__init__(
            name,
            framework=SupportedFrameworks.SMOLAGENTS,
            host=host,
            port=port,
            max_iterations=max_iterations or agent.max_steps,
            state_store=state_store,
        )

        # Register metadata
        self._register_agent_metadata(
            agent=self._agent,
            framework=SupportedFrameworks.SMOLAGENTS,
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
            call_model_activity, name="smolagents_call_model_activity"
        )
        self._workflow_runtime.register_activity(
            execute_tool_activity, name="smolagents_execute_tool_activity"
        )

    def _register_model_and_tools(self) -> None:
        """Register the agent's model and tools in the global process registry."""
        clear_registries()
        register_model(self._sanitized_name, self._agent.model)
        for tool_name, tool in self._agent.tools.items():
            register_tool(tool_name, tool)
            logger.info("Registered tool: %s", tool_name)

    def _get_agent_config(self) -> AgentConfig:
        """Extract serializable agent configuration."""
        tool_definitions = [
            ToolDefinition(
                name=tool_name,
                description=getattr(tool, "description", "") or "",
                inputs=getattr(tool, "inputs", {}) or {},
                output_type=getattr(tool, "output_type", "string"),
            )
            for tool_name, tool in self._agent.tools.items()
        ]

        return AgentConfig(
            name=self._name,
            model_key=self._sanitized_name,
            tool_definitions=tool_definitions,
        )

    # ------------------------------------------------------------------
    # Framework-specific run methods
    # ------------------------------------------------------------------

    def _build_agent_workflow_input(
        self,
        task: str,
        session_id: str,
        agent_config: AgentConfig,
    ) -> dict[str, Any]:
        """Build the AgentWorkflowInput payload shared by run_async and the
        serve default factory (registered in _setup_serve_defaults)."""
        messages = [
            ChatEntry(role="system", content=self._system_prompt or ""),
            ChatEntry(role="user", content=f"New task:\n{task}"),
        ]

        return AgentWorkflowInput(
            agent_config=agent_config,
            messages=messages,
            session_id=session_id,
            iteration=0,
            max_iterations=self._max_iterations,
        ).to_dict()

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
            workflow_id = f"smolagents-{session_id}-{uuid.uuid4().hex[:8]}"

        workflow_input_dict = self._build_agent_workflow_input(
            task, session_id, self._get_agent_config()
        )
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
                "result": output.final_answer,
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
                        final_answer=event.get("result"),
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
        setup_telemetry(self.__class__.__name__, config=self._observability_config)
        instrument_grpc(config=self._observability_config)

    def _setup_serve_defaults(self) -> None:
        agent_config = self._get_agent_config()

        def _build_workflow_input(task_str: str) -> dict[str, Any]:
            return self._build_agent_workflow_input(
                task_str, uuid.uuid4().hex[:8], agent_config
            )

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
    def agent(self) -> "ToolCallingAgent":
        """The smolagents agent being executed."""
        return self._agent
