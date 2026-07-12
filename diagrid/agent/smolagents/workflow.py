# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr Workflow definitions for durable smolagents ``ToolCallingAgent`` execution."""

import json
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
    CallModelInput,
    CallModelOutput,
    ChatEntry,
    ExecuteToolInput,
    ExecuteToolOutput,
    ToolCall,
)

logger = logging.getLogger(__name__)

FINAL_ANSWER_TOOL_NAME = "final_answer"
"""smolagents auto-injects this tool and terminates on a bare name match
(``agents.py``: ``is_final_answer = tool_name == "final_answer"``) — not an
isinstance/type check. We mirror that exact string-match convention."""


# Global registries - populated by the runner. A single active model/tool set
# per process, matching the simplification already used by the ADK and
# LangChain integrations.
_model_registry: dict[str, Any] = {}
_tool_registry: dict[str, Any] = {}
_default_workflow_input_factory: Optional[Callable[[str], dict[str, Any]]] = None


def register_model(key: str, model: Any) -> None:
    """Register a smolagents ``Model`` instance for use by the call_model activity."""
    _model_registry[key] = model


def get_registered_model(key: str) -> Optional[Any]:
    """Get a registered model by key."""
    return _model_registry.get(key)


def register_tool(name: str, tool: Any) -> None:
    """Register a smolagents ``Tool`` for use by the execute_tool activity."""
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


def _format_tool_calls_text(tool_calls: list[ToolCall]) -> str:
    """Render a "Calling tools:" turn, mirroring smolagents' own formatting."""
    calls = [{"name": tc.name, "arguments": tc.args} for tc in tool_calls]
    return "Calling tools:\n" + json.dumps(calls)


def _format_observation_text(outputs: list[ExecuteToolOutput]) -> str:
    """Render an "Observation:" turn from tool results, mirroring smolagents."""
    lines = [f"- {o.tool_name}: {o.content}" for o in outputs]
    return "Observation:\n" + "\n".join(lines)


def agent_workflow(
    ctx: DaprWorkflowContext, input_data: dict[str, Any]
) -> Generator[Any, Any, Any]:
    """Dapr Workflow that orchestrates a smolagents ``ToolCallingAgent`` loop.

    This workflow:
    1. Calls the model to get the next action (as an activity)
    2. If the model calls ``final_answer``, terminates with that argument
    3. Otherwise executes each requested tool (as separate activities) and
       loops back until ``final_answer`` is called or ``max_iterations`` is hit

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
        model_input = CallModelInput(
            agent_config=workflow_input.agent_config,
            messages=workflow_input.messages,
        )

        model_output_data = yield ctx.call_activity(
            call_model_activity,
            input=model_input.to_dict(),
            retry_policy=retry_policy,
        )
        model_output = CallModelOutput.from_dict(model_output_data)

        if model_output.error:
            return AgentWorkflowOutput(
                final_answer=None,
                messages=workflow_input.messages,
                iterations=iteration,
                status="error",
                error=model_output.error,
            ).to_dict()

        tool_calls = model_output.tool_calls

        if not tool_calls:
            # tool_choice is forced ("required") whenever tools are passed, so
            # smolagents always returns at least one tool call in practice.
            # Defensively treat a content-only response as the final answer.
            return AgentWorkflowOutput(
                final_answer=model_output.content,
                messages=workflow_input.messages,
                iterations=iteration + 1,
                status="completed",
            ).to_dict()

        workflow_input.messages.append(
            ChatEntry(role="assistant", content=_format_tool_calls_text(tool_calls))
        )

        final_call = next(
            (tc for tc in tool_calls if tc.name == FINAL_ANSWER_TOOL_NAME), None
        )
        if final_call is not None:
            answer = final_call.args.get("answer")
            return AgentWorkflowOutput(
                final_answer=str(answer) if answer is not None else None,
                messages=workflow_input.messages,
                iterations=iteration + 1,
                status="completed",
            ).to_dict()

        # Execute each tool call as a separate activity, in parallel.
        tool_tasks = []
        for tool_call in tool_calls:
            tool_input = ExecuteToolInput(tool_call=tool_call)
            task = ctx.call_activity(
                execute_tool_activity,
                input=tool_input.to_dict(),
                retry_policy=retry_policy,
            )
            tool_tasks.append(task)

        tool_outputs_data = yield when_all(tool_tasks)
        tool_outputs = [ExecuteToolOutput.from_dict(o) for o in tool_outputs_data]

        workflow_input.messages.append(
            ChatEntry(role="user", content=_format_observation_text(tool_outputs))
        )

        iteration += 1

    return AgentWorkflowOutput(
        final_answer=None,
        messages=workflow_input.messages,
        iterations=iteration,
        status="max_iterations_reached",
        error=f"Max iterations ({workflow_input.max_iterations}) reached",
    ).to_dict()


def _coerce_args(raw: Any) -> dict[str, Any]:
    """Normalize tool-call arguments to a dict.

    ``ChatMessageToolCallFunction.arguments`` is a JSON string straight off
    the wire; smolagents itself parses it with ``parse_json_if_needed``
    before executing. We do the equivalent here.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def call_model_activity(
    ctx: WorkflowActivityContext, input_data: dict[str, Any]
) -> dict[str, Any]:
    """Activity that calls the smolagents model.

    Exceptions are intentionally left to propagate (after logging) so that
    Dapr's retry policy on this activity governs transient failures.

    Args:
        ctx: The workflow activity context
        input_data: Dictionary containing CallModelInput data

    Returns:
        CallModelOutput as a dictionary
    """
    model_input = CallModelInput.from_dict(input_data)

    from diagrid.agent.core.telemetry import get_tracer

    _tracer = get_tracer("smolagents.agent")
    _span = _tracer.start_span("Model.generate") if _tracer else None

    try:
        model = get_registered_model(model_input.agent_config.model_key)
        if model is None:
            return CallModelOutput(
                content=None,
                tool_calls=[],
                error=(
                    f"No model registered under key "
                    f"'{model_input.agent_config.model_key}'"
                ),
            ).to_dict()

        tools = [
            get_registered_tool(td.name)
            for td in model_input.agent_config.tool_definitions
        ]
        tools = [t for t in tools if t is not None]

        messages = [m.to_model_message() for m in model_input.messages]
        if _span:
            _span.set_attribute("llm.message_count", len(messages))

        chat_message = model.generate(messages, tools_to_call_from=tools)

        tool_calls = [
            ToolCall(
                id=str(tc.id),
                name=str(tc.function.name),
                args=_coerce_args(tc.function.arguments),
            )
            for tc in (chat_message.tool_calls or [])
        ]

        content = (
            chat_message.content if isinstance(chat_message.content, str) else None
        )

        return CallModelOutput(content=content, tool_calls=tool_calls).to_dict()

    except Exception:
        if _span:
            _span.set_attribute("error", True)
        logger.exception("Error calling smolagents model")
        raise
    finally:
        if _span:
            _span.end()


def execute_tool_activity(
    ctx: WorkflowActivityContext, input_data: dict[str, Any]
) -> dict[str, Any]:
    """Activity that executes a single smolagents tool.

    Only an unknown tool name is treated as a non-retryable, immediately
    returned error. Any exception raised by the tool itself propagates so
    Dapr's retry policy applies.

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

    _tracer = get_tracer("smolagents.agent")
    _span = _tracer.start_span("Tool.call") if _tracer else None
    if _span:
        _span.set_attribute("tool.name", tool_call.name)

    try:
        result = tool(**tool_call.args, sanitize_inputs_outputs=True)
        content = result if isinstance(result, str) else str(result)
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
