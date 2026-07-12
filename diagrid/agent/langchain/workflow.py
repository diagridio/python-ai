# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr Workflow definitions for durable LangChain agent execution."""

import logging
from datetime import timedelta
from typing import Any, Callable, Generator, Optional

from dapr.ext.workflow import (
    DaprWorkflowContext,
    WorkflowActivityContext,
    RetryPolicy,
    when_all,
)

from .models import (
    AgentWorkflowInput,
    AgentWorkflowOutput,
    CallLlmInput,
    CallLlmOutput,
    ExecuteToolInput,
    ExecuteToolOutput,
    Message,
    ToolCall,
)

logger = logging.getLogger(__name__)


# Global registries - populated by the runner. A single active model/tool set
# per process, matching the simplification already used by the ADK and
# Strands integrations (see their respective ``workflow.py``/``runner.py``).
_model_registry: dict[str, Any] = {}
_tool_registry: dict[str, Any] = {}
_default_workflow_input_factory: Optional[Callable[[str], dict[str, Any]]] = None


def register_model(key: str, model: Any) -> None:
    """Register a ``BaseChatModel`` instance for use by the call_llm activity."""
    _model_registry[key] = model


def get_registered_model(key: str) -> Optional[Any]:
    """Get a registered model by key."""
    return _model_registry.get(key)


def register_tool(name: str, tool: Any) -> None:
    """Register a ``BaseTool`` for use by the execute_tool activity."""
    _tool_registry[name] = tool


def get_registered_tool(name: str) -> Optional[Any]:
    """Get a registered tool by name."""
    return _tool_registry.get(name)


def set_default_workflow_input_factory(
    factory: Callable[[str], dict[str, Any]],
) -> None:
    """Store a factory that builds AgentWorkflowInput dicts from a task string."""
    global _default_workflow_input_factory
    _default_workflow_input_factory = factory


def clear_registries() -> None:
    """Clear all registered models and tools."""
    global _default_workflow_input_factory
    _model_registry.clear()
    _tool_registry.clear()
    _default_workflow_input_factory = None


def agent_workflow(
    ctx: DaprWorkflowContext, input_data: dict[str, Any]
) -> Generator[Any, Any, Any]:
    """Dapr Workflow that orchestrates a LangChain model + tools agent loop.

    This workflow:
    1. Calls the chat model to get the next action (as an activity)
    2. If the model returns tool calls, executes each tool (as separate activities)
    3. Loops back until the model returns a final response with no tool calls

    All iterations run within a single workflow instance.

    Args:
        ctx: The Dapr workflow context
        input_data: Dictionary containing AgentWorkflowInput data

    Returns:
        AgentWorkflowOutput as a dictionary
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

    iteration = 0

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

        # Execute each tool call as a separate activity, in parallel.
        tool_tasks = []
        for tool_call in llm_output.message.tool_calls:
            tool_input = ExecuteToolInput(tool_call=tool_call)
            task = ctx.call_activity(
                execute_tool_activity,
                input=tool_input.to_dict(),
                retry_policy=retry_policy,
            )
            tool_tasks.append(task)

        tool_outputs_data = yield when_all(tool_tasks)

        # Each tool result becomes its own tool-role message, in the same
        # order as the requesting tool_calls (mirrors langchain_core's
        # ToolMessage-per-call convention).
        for tool_output_data in tool_outputs_data:
            tool_output = ExecuteToolOutput.from_dict(tool_output_data)
            workflow_input.messages.append(
                Message(
                    role="tool",
                    content=tool_output.content,
                    tool_call_id=tool_output.tool_call_id,
                    name=tool_output.tool_name,
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


def _to_langchain_messages(messages: list[Message]) -> list[Any]:
    """Convert serializable ``Message`` objects to real ``langchain_core`` messages."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    lc_messages: list[Any] = []
    for msg in messages:
        if msg.role == "system":
            lc_messages.append(SystemMessage(content=msg.content or ""))
        elif msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content or ""))
        elif msg.role == "assistant":
            lc_messages.append(
                AIMessage(
                    content=msg.content or "",
                    tool_calls=[tc.to_langchain_tool_call() for tc in msg.tool_calls],
                )
            )
        elif msg.role == "tool":
            lc_messages.append(
                ToolMessage(
                    content=msg.content or "",
                    tool_call_id=msg.tool_call_id or "",
                    name=msg.name,
                )
            )
        else:
            raise ValueError(f"Unknown message role: {msg.role}")
    return lc_messages


def call_llm_activity(
    ctx: WorkflowActivityContext, input_data: dict[str, Any]
) -> dict[str, Any]:
    """Activity that calls the LangChain chat model.

    Exceptions are intentionally left to propagate (after logging) so that
    Dapr's retry policy on this activity governs transient failures — the
    same pattern used by the ADK integration's ``call_llm_activity``.

    Args:
        ctx: The workflow activity context
        input_data: Dictionary containing CallLlmInput data

    Returns:
        CallLlmOutput as a dictionary
    """
    llm_input = CallLlmInput.from_dict(input_data)

    from diagrid.agent.core.telemetry import get_tracer

    _tracer = get_tracer("langchain.agent")
    _span = _tracer.start_span("ChatModel.invoke") if _tracer else None

    try:
        model = get_registered_model(llm_input.agent_config.model_key)
        if model is None:
            return CallLlmOutput(
                message=Message(role="assistant", content=None),
                is_final=True,
                error=(
                    f"No model registered under key "
                    f"'{llm_input.agent_config.model_key}'"
                ),
            ).to_dict()

        tools = [
            get_registered_tool(td.name)
            for td in llm_input.agent_config.tool_definitions
        ]
        tools = [t for t in tools if t is not None]

        model_with_tools = model.bind_tools(tools) if tools else model

        lc_messages = _to_langchain_messages(llm_input.messages)
        if _span:
            _span.set_attribute("llm.message_count", len(lc_messages))

        ai_message = model_with_tools.invoke(lc_messages)

        tool_calls = [
            ToolCall(
                id=str(tc.get("id") or ""),
                name=str(tc.get("name") or ""),
                args=dict(tc.get("args") or {}),
            )
            for tc in (ai_message.tool_calls or [])
        ]

        content = ai_message.content if isinstance(ai_message.content, str) else None

        return CallLlmOutput(
            message=Message(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            ),
            is_final=len(tool_calls) == 0,
        ).to_dict()

    except Exception:
        if _span:
            _span.set_attribute("error", True)
        logger.exception("Error calling LangChain chat model")
        raise
    finally:
        if _span:
            _span.end()


def execute_tool_activity(
    ctx: WorkflowActivityContext, input_data: dict[str, Any]
) -> dict[str, Any]:
    """Activity that executes a single LangChain tool.

    Only an unknown tool name is treated as a non-retryable, immediately
    returned error (retrying can't help). Any exception raised by the tool
    itself propagates so Dapr's retry policy applies — this is what makes
    the crash-recovery / retry examples actually exercise Dapr's retry
    mechanism rather than an application-level catch-all.

    Args:
        ctx: The workflow activity context
        input_data: Dictionary containing ExecuteToolInput data

    Returns:
        ExecuteToolOutput as a dictionary
    """
    tool_input = ExecuteToolInput.from_dict(input_data)
    tool_call = tool_input.tool_call

    tool = get_registered_tool(tool_call.name)
    if tool is None:
        return ExecuteToolOutput(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=f"Error: unknown tool '{tool_call.name}'",
            is_error=True,
        ).to_dict()

    from diagrid.agent.core.telemetry import get_tracer

    _tracer = get_tracer("langchain.agent")
    _span = _tracer.start_span("Tool.invoke") if _tracer else None
    if _span:
        _span.set_attribute("tool.name", tool_call.name)

    try:
        tool_message = tool.invoke(tool_call.to_langchain_tool_call())
        content = tool_message.content
        if not isinstance(content, str):
            content = str(content)
        return ExecuteToolOutput(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=content,
        ).to_dict()
    except Exception:
        if _span:
            _span.set_attribute("error", True)
        logger.exception("Error executing tool '%s'", tool_call.name)
        raise
    finally:
        if _span:
            _span.end()
