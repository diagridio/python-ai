# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr Workflow definitions for durable Claude Agent SDK execution.

Each agent invocation runs as a Dapr workflow. Within that workflow, every LLM
turn is dispatched as a ``call_llm_activity`` and every tool invocation as a
separate ``execute_tool_activity`` — giving each step its own durable retry,
checkpoint, and replay boundary.
"""

import json
import logging
from datetime import timedelta
from typing import Any, Callable, Generator, Optional

from dapr.ext.workflow import (
    DaprWorkflowContext,
    RetryPolicy,
    WorkflowActivityContext,
    when_all,
)

from .models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    CallLlmInput,
    CallLlmOutput,
    ExecuteToolInput,
    ExecuteToolOutput,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)


# Global tool registry — tools are registered by the runner.
# Activities run in the same process as the runner, so this in-process registry
# is reachable from execute_tool_activity. Tool *definitions* (schemas) travel
# through workflow inputs; only the executable callable stays in this registry.
_tool_registry: dict[str, Any] = {}
_tool_definitions: dict[str, ToolDefinition] = {}
_default_workflow_input_factory: Optional[Callable[[str], dict[str, Any]]] = None


def register_tool(
    name: str, tool: Any, definition: Optional[ToolDefinition] = None
) -> None:
    """Register a tool callable for use by the execute_tool activity."""
    _tool_registry[name] = tool
    if definition:
        _tool_definitions[name] = definition


def get_registered_tool(name: str) -> Optional[Any]:
    """Get a registered tool by name."""
    return _tool_registry.get(name)


def get_tool_definition(name: str) -> Optional[ToolDefinition]:
    """Get a tool definition by name."""
    return _tool_definitions.get(name)


def set_default_workflow_input_factory(
    factory: Callable[[str], dict[str, Any]],
) -> None:
    """Store a factory that builds AgentWorkflowInput dicts from a task string."""
    global _default_workflow_input_factory
    _default_workflow_input_factory = factory


def clear_tool_registry() -> None:
    """Clear all registered tools and the default workflow input factory."""
    global _default_workflow_input_factory
    _tool_registry.clear()
    _tool_definitions.clear()
    _default_workflow_input_factory = None


def agent_workflow(
    ctx: DaprWorkflowContext, input_data: dict[str, Any]
) -> Generator[Any, Any, Any]:
    """Dapr workflow orchestrating a Claude Agent SDK execution.

    The workflow drives the agent loop directly: each iteration calls the LLM
    as one activity, then fans tool calls out as parallel activities, then
    feeds the results back into the next LLM activity. This keeps every LLM
    call and every tool call as a separate durable Dapr activity.
    """
    if "agent_config" in input_data:
        workflow_input = AgentWorkflowInput.from_dict(input_data)
    elif _default_workflow_input_factory is not None:
        task = input_data.get("task", "")
        workflow_input = AgentWorkflowInput.from_dict(
            _default_workflow_input_factory(task)
        )
    else:
        raise ValueError(
            "Received input without 'agent_config' and no default factory is set. "
            "Ensure the runner has been started before the workflow is invoked."
        )

    retry_policy = RetryPolicy(
        max_number_of_attempts=3,
        first_retry_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
        max_retry_interval=timedelta(seconds=30),
    )

    iteration = workflow_input.iteration

    while iteration < workflow_input.max_iterations:
        llm_input = CallLlmInput(
            agent_config=workflow_input.agent_config,
            messages=workflow_input.messages,
        )

        llm_output_data = yield ctx.call_activity(
            call_llm_activity,
            input=llm_input.to_dict(),
            retry_policy=retry_policy,
        )
        llm_output = CallLlmOutput.from_dict(llm_output_data)

        if llm_output.error:
            return AgentWorkflowOutput(
                final_response=None,
                messages=workflow_input.messages,
                iterations=iteration,
                status="error",
                error=llm_output.error,
            ).to_dict()

        workflow_input.messages.append(llm_output.message)

        if llm_output.is_final:
            return AgentWorkflowOutput(
                final_response=llm_output.message.content,
                messages=workflow_input.messages,
                iterations=iteration + 1,
                status="completed",
            ).to_dict()

        # Fan tool calls out as parallel activities — each one is its own
        # durable boundary so the workflow can resume mid-batch on failure.
        tool_tasks = []
        for tool_call in llm_output.message.tool_calls:
            tool_input = ExecuteToolInput(
                tool_call=tool_call,
                agent_name=workflow_input.agent_config.name,
                session_id=workflow_input.session_id,
            )
            task = ctx.call_activity(
                execute_tool_activity,
                input=tool_input.to_dict(),
                retry_policy=retry_policy,
            )
            tool_tasks.append((tool_call, task))

        tool_output_tasks = [t for _, t in tool_tasks]
        tool_outputs_data = yield when_all(tool_output_tasks)

        for tool_output_data in tool_outputs_data:
            tool_output = ExecuteToolOutput.from_dict(tool_output_data)
            tr = tool_output.tool_result
            if tr.error:
                content = f"Error: {tr.error}"
            elif tr.result is not None:
                content = str(tr.result)
            else:
                content = ""
            workflow_input.messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=content,
                    tool_call_id=tr.tool_call_id,
                    name=tr.tool_name,
                )
            )

        iteration += 1

    return AgentWorkflowOutput(
        final_response=None,
        messages=workflow_input.messages,
        iterations=iteration,
        status="max_iterations_reached",
        error=f"Max iterations ({workflow_input.max_iterations}) reached",
    ).to_dict()


def call_llm_activity(
    ctx: WorkflowActivityContext, input_data: dict[str, Any]
) -> dict[str, Any]:
    """Activity that performs one call to the Anthropic Messages API.

    This is the per-LLM-call durable boundary: one invocation produces exactly
    one assistant turn (text and/or tool_use blocks), which the workflow then
    inspects to decide whether to fan out tool activities or return.
    """
    llm_input = CallLlmInput.from_dict(input_data)

    try:
        import anthropic

        anthropic_messages = _build_anthropic_messages(llm_input.messages)
        anthropic_tools = _build_anthropic_tools(
            llm_input.agent_config.tool_definitions
        )

        from diagrid.agent.core.telemetry import get_tracer

        _tracer = get_tracer("claude.agent")
        _span = _tracer.start_span("LLM.messages_create") if _tracer else None
        if _span:
            _span.set_attribute("llm.model", llm_input.agent_config.model)
        try:
            client = anthropic.Anthropic()
            kwargs: dict[str, Any] = {
                "model": llm_input.agent_config.model,
                "max_tokens": llm_input.agent_config.max_tokens,
                "messages": anthropic_messages,
            }
            if llm_input.agent_config.system_prompt:
                kwargs["system"] = llm_input.agent_config.system_prompt
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools

            response = client.messages.create(**kwargs)
        except Exception:
            if _span:
                _span.set_attribute("error", True)
            raise
        finally:
            if _span:
                _span.end()

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        args=dict(getattr(block, "input", {}) or {}),
                    )
                )

        stop_reason = getattr(response, "stop_reason", None)
        output_message = Message(
            role=MessageRole.ASSISTANT,
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
        )

        is_final = stop_reason == "end_turn" and not tool_calls

        return CallLlmOutput(
            message=output_message,
            is_final=is_final,
            stop_reason=stop_reason,
        ).to_dict()

    except Exception as e:
        logger.error("Error calling Claude: %s", e)
        import traceback

        traceback.print_exc()
        return CallLlmOutput(
            message=Message(role=MessageRole.ASSISTANT),
            is_final=True,
            error=str(e),
        ).to_dict()


def execute_tool_activity(
    ctx: WorkflowActivityContext, input_data: dict[str, Any]
) -> dict[str, Any]:
    """Activity that executes a single tool call.

    Looks up the tool by name in the in-process registry, invokes it, and
    returns the serialized result so the workflow can fold it back into the
    next LLM activity's message history.
    """
    tool_input = ExecuteToolInput.from_dict(input_data)
    tool_call = tool_input.tool_call

    tool = get_registered_tool(tool_call.name)
    if tool is None:
        return ExecuteToolOutput(
            tool_result=ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=None,
                error=f"Tool '{tool_call.name}' not found in registry",
            )
        ).to_dict()

    try:
        result = _execute_tool(tool, tool_call.args)
        result = _serialize_tool_result(result)
        return ExecuteToolOutput(
            tool_result=ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
            )
        ).to_dict()

    except Exception as e:
        logger.error("Error executing tool '%s': %s", tool_call.name, e)
        import traceback

        traceback.print_exc()
        return ExecuteToolOutput(
            tool_result=ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=None,
                error=str(e),
            )
        ).to_dict()


def _build_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert workflow Messages to Anthropic Messages API format.

    Anthropic merges consecutive tool_result blocks into a single user message,
    so we group sequential MessageRole.TOOL entries into one user turn.
    """
    result: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            result.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for msg in messages:
        if msg.role == MessageRole.TOOL:
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content or "",
                }
            )
            continue

        flush_tool_results()

        if msg.role == MessageRole.USER:
            result.append({"role": "user", "content": msg.content or ""})
        elif msg.role == MessageRole.ASSISTANT:
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.args,
                    }
                )
            result.append({"role": "assistant", "content": blocks})

    flush_tool_results()
    return result


def _build_anthropic_tools(
    tool_defs: list[ToolDefinition],
) -> list[dict[str, Any]]:
    """Convert ToolDefinition list to Anthropic tools format."""
    tools = []
    for td in tool_defs:
        schema = td.parameters or {"type": "object", "properties": {}}
        tools.append(
            {
                "name": td.name,
                "description": td.description,
                "input_schema": schema,
            }
        )
    return tools


def _execute_tool(tool: Any, args: dict[str, Any]) -> Any:
    """Invoke a registered tool callable.

    Supports:
        - Plain Python functions (sync or async)
        - SdkMcpTool instances (from ``claude_agent_sdk.tool`` decorator)
        - Anything else with a ``handler`` async coroutine
    """
    import asyncio

    handler = getattr(tool, "handler", None)
    if handler is not None:
        coro = handler(args)
        if asyncio.iscoroutine(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        return coro

    if callable(tool):
        result = tool(**args)
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(result)
            finally:
                loop.close()
        return result

    raise TypeError(f"Tool {tool!r} is not callable and has no handler attribute")


def _serialize_tool_result(result: Any) -> Any:
    """Convert a tool result to a JSON-serializable form for the workflow."""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, (str, int, float, bool, list, dict, type(None))):
        return result
    try:
        return json.loads(json.dumps(result))
    except (TypeError, ValueError):
        return str(result)
