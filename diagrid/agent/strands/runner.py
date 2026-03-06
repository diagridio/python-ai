# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Runner for executing Strands agents as Dapr Workflows."""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Generator, Optional, TYPE_CHECKING

from diagrid.agent.core.chat import (
    ChatMessage,
    ChatRole,
    ChatToolCall,
    ChatToolDefinition,
    get_chat_client,
)
from diagrid.agent.core.workflow import BaseWorkflowRunner

from .workflow import WorkflowOutput

if TYPE_CHECKING:
    from strands import Agent

logger = logging.getLogger(__name__)


class DaprWorkflowAgentRunner(BaseWorkflowRunner):
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
        component_name: Optional[str] = None,
        state_store: Optional[Any] = None,
    ):
        """Initialize the runner.

        Args:
            agent: The Strands Agent to execute
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
            host=host,
            port=port,
            max_iterations=max_iterations or 25,
            component_name=component_name,
            state_store=state_store,
        )

        # Auto-detect: if user provided no model on the agent, resolve a component
        if self._component_name is None and not self._agent_has_model():
            self._dapr_chat_client = get_chat_client()
            self._component_name = self._dapr_chat_client.component_name
            logger.info(
                "No model configured on agent; using Dapr conversation component: %s",
                self._component_name,
            )
        elif self._component_name is not None:
            self._dapr_chat_client = get_chat_client(self._component_name)

        # Register metadata
        self._register_agent_metadata(
            agent=self._agent,
            framework="strands",
            registry=registry_config,
            component_name=self._component_name,
            state_store_name=self._state_store.store_name
            if self._state_store
            else None,
        )

        self._register_workflow_components()

    def _agent_has_model(self) -> bool:
        """Check if the agent has an explicit model configured."""
        model = getattr(self._agent, "model", None)
        return model is not None

    def _register_workflow_components(self) -> None:
        """Set up the workflow runtime, activities, and workflow."""
        # Store references for closures
        agent_ref = self._agent
        component_name_ref = self._component_name

        # ============================================================
        # Activity: Call the model (LLM)
        # ============================================================
        def _call_model_via_dapr(messages: list, system_prompt: str) -> dict:  # type: ignore[type-arg]
            """Route model call through Dapr Conversation API."""
            chat_messages = []
            if system_prompt:
                chat_messages.append(
                    ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)
                )

            for msg in messages:
                role = msg.get("role", "user")
                content_blocks = msg.get("content", [])

                if role == "user":
                    has_tool_results = False
                    for block in content_blocks:
                        if isinstance(block, dict) and "toolResult" in block:
                            has_tool_results = True
                            tr = block["toolResult"]
                            result_text = ""
                            for c in tr.get("content", []):
                                if isinstance(c, dict) and "text" in c:
                                    result_text += c["text"]
                            chat_messages.append(
                                ChatMessage(
                                    role=ChatRole.TOOL,
                                    content=result_text,
                                    tool_call_id=tr.get("toolUseId", ""),
                                    name="",
                                )
                            )
                    if not has_tool_results:
                        text_parts = []
                        for block in content_blocks:
                            if isinstance(block, dict) and "text" in block:
                                text_parts.append(block["text"])
                        chat_messages.append(
                            ChatMessage(
                                role=ChatRole.USER,
                                content="".join(text_parts),
                            )
                        )
                elif role == "assistant":
                    tool_calls = []
                    text_parts = []
                    for block in content_blocks:
                        if isinstance(block, dict):
                            if "toolUse" in block:
                                tu = block["toolUse"]
                                tool_calls.append(
                                    ChatToolCall(
                                        id=tu.get("toolUseId", ""),
                                        name=tu.get("name", ""),
                                        arguments=json.dumps(tu.get("input", {})),
                                    )
                                )
                            elif "text" in block:
                                text_parts.append(block["text"])
                    chat_messages.append(
                        ChatMessage(
                            role=ChatRole.ASSISTANT,
                            content="".join(text_parts) if text_parts else None,
                            tool_calls=tool_calls,
                        )
                    )

            # Build tool definitions from agent's tool registry
            tools = None
            tool_specs = [
                tool.tool_spec for tool in agent_ref.tool_registry.registry.values()
            ]
            if tool_specs:
                tools = []
                for spec in tool_specs:
                    tools.append(
                        ChatToolDefinition(
                            name=spec.get("name", ""),
                            description=spec.get("description", ""),
                            parameters=spec.get("inputSchema"),
                        )
                    )

            client = get_chat_client(component_name_ref)
            response = client.chat(messages=chat_messages, tools=tools or None)

            # Convert back to Strands format
            content_out: list[Any] = []
            tool_uses: list[Any] = []

            if response.content:
                content_out.append({"text": response.content})

            for tc in response.tool_calls:
                try:
                    args = json.loads(tc.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tu = {
                    "name": tc.name,
                    "toolUseId": tc.id,
                    "input": args,
                }
                tool_uses.append(tu)
                content_out.append({"toolUse": tu})

            stop_reason = "tool_use" if tool_uses else "end_turn"

            return {
                "tool_uses": tool_uses,
                "text": response.content or "",
                "stop_reason": stop_reason,
                "content": content_out,
            }

        def call_model_activity(ctx: Any, input_data: dict) -> dict:  # type: ignore[type-arg]
            """Activity that calls the LLM model."""
            messages = input_data.get("messages", [])
            system_prompt = input_data.get("system_prompt", "")

            # Route through Dapr if component_name is set
            if component_name_ref:
                return _call_model_via_dapr(messages, system_prompt)

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
            """Workflow that orchestrates the agent loop."""
            task = input_data.get("task", "")
            system_prompt = agent_ref.system_prompt or ""

            messages: list[dict[str, Any]] = [
                {"role": "user", "content": [{"text": task}]}
            ]
            final_text = ""
            all_tool_calls: list[dict[str, Any]] = []

            for iteration in range(max_iters):
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

                messages.append({"role": "assistant", "content": content})

                if not tool_uses or stop_reason == "end_turn":
                    final_text = text
                    break

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

                    result = yield ctx.call_activity(
                        execute_tool_activity,
                        input={
                            "tool_name": t_name,
                            "tool_use_id": t_id,
                            "tool_input": t_input,
                        },
                    )
                    tool_results.append({"toolResult": result})

                messages.append({"role": "user", "content": tool_results})

            return {  # type: ignore[return-value]
                "result": final_text,
                "tool_calls": all_tool_calls,
            }

        # Register workflow and activities
        self._workflow_runtime.register_workflow(agent_workflow, name="agent_workflow")
        self._workflow_runtime.register_activity(
            call_model_activity, name="strands_call_model"
        )
        self._workflow_runtime.register_activity(
            execute_tool_activity, name="strands_execute_tool"
        )

        # Store reference
        self._workflow_func = agent_workflow

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
            workflow_id = f"strands-{session_id}-{uuid.uuid4().hex[:8]}"

        input_data = {"task": task}

        logger.info("Starting workflow: %s", workflow_id)
        self._workflow_client.schedule_new_workflow(
            workflow=self._workflow_func,
            input=input_data,
            instance_id=workflow_id,
        )

        yield {
            "type": "workflow_started",
            "workflow_id": workflow_id,
            "session_id": session_id,
        }

        def _parse_output(wf_id: str, output_dict: dict) -> dict:  # type: ignore[type-arg]
            return {
                "type": "workflow_completed",
                "workflow_id": wf_id,
                "result": output_dict.get("result", ""),
                "tool_calls": output_dict.get("tool_calls", []),
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
    ) -> WorkflowOutput:
        """Run the agent synchronously and wait for completion.

        Args:
            task: The user task to send to the agent
            session_id: Session ID for the execution
            workflow_id: Optional workflow instance ID (generated if not provided)
            timeout: Maximum time to wait for completion (seconds)

        Returns:
            WorkflowOutput with the agent's response
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

        return self._run_sync(_run(), timeout=timeout)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Serve overrides
    # ------------------------------------------------------------------

    def _setup_telemetry(self) -> None:
        from diagrid.agent.core.telemetry import _make_span_processor, instrument_grpc

        try:
            from strands.telemetry.config import StrandsTelemetry

            st = StrandsTelemetry()
            processor = _make_span_processor(config=self._observability_config)
            if processor is not None:
                st.tracer_provider.add_span_processor(processor)
        except Exception:
            logger.debug("Strands native OTEL setup skipped", exc_info=True)

        instrument_grpc(config=self._observability_config)

    def _setup_serve_defaults(self) -> None:
        pass  # Strands doesn't use a workflow input factory

    async def _serve_run(
        self,
        request: dict,
        session_id: str,  # type: ignore[type-arg]
    ) -> AsyncIterator[dict[str, Any]]:
        task = request.get("task") or ""
        async for event in self.run_async(task=task, session_id=session_id):
            yield event

    @property
    def agent(self) -> "Agent":
        """The Strands agent being executed."""
        return self._agent
