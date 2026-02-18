# Copyright 2025 Diagrid Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dapr Workflow definitions for durable OpenAI Agents SDK execution."""

import json
import logging
from datetime import timedelta
from typing import Any, Generator, Optional

from dapr.ext.workflow import (
    DaprWorkflowContext,
    WorkflowActivityContext,
    RetryPolicy,
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
    ToolResult,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


# Global tool registry - tools are registered by the runner
_tool_registry: dict[str, Any] = {}
_tool_definitions: dict[str, ToolDefinition] = {}


def register_tool(
    name: str, tool: Any, definition: Optional[ToolDefinition] = None
) -> None:
    """Register a tool for use by the execute_tool activity.

    Args:
        name: The tool name
        tool: The actual tool object (FunctionTool or callable)
        definition: Optional serializable tool definition
    """
    _tool_registry[name] = tool
    if definition:
        _tool_definitions[name] = definition


def get_registered_tool(name: str) -> Optional[Any]:
    """Get a registered tool by name."""
    return _tool_registry.get(name)


def get_tool_definition(name: str) -> Optional[ToolDefinition]:
    """Get a tool definition by name."""
    return _tool_definitions.get(name)


def clear_tool_registry() -> None:
    """Clear all registered tools."""
    _tool_registry.clear()
    _tool_definitions.clear()


def openai_agents_workflow(
    ctx: DaprWorkflowContext, input_data: dict[str, Any]
) -> Generator[Any, Any, Any]:
    """Dapr Workflow that orchestrates an OpenAI Agents SDK agent execution.

    This workflow:
    1. Calls the LLM to get the next action (as an activity)
    2. If the LLM returns tool calls, executes each tool in parallel (as separate activities)
    3. Loops back until the LLM returns a final response

    All iterations run within a single workflow instance, making the entire
    agent execution durable and resumable.

    Args:
        ctx: The Dapr workflow context
        input_data: Dictionary containing AgentWorkflowInput data

    Returns:
        AgentWorkflowOutput as a dictionary
    """
    # Deserialize input
    workflow_input = AgentWorkflowInput.from_dict(input_data)

    # Retry policy for activities
    retry_policy = RetryPolicy(
        max_number_of_attempts=3,
        first_retry_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
        max_retry_interval=timedelta(seconds=30),
    )

    iteration = workflow_input.iteration

    # Main agent loop - runs until final response or max iterations
    while iteration < workflow_input.max_iterations:
        # Activity: Call LLM to get next action
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

        # Handle LLM errors
        if llm_output.error:
            return AgentWorkflowOutput(
                final_response=None,
                messages=workflow_input.messages,
                iterations=iteration,
                status="error",
                error=llm_output.error,
            ).to_dict()

        # Add LLM response to messages
        workflow_input.messages.append(llm_output.message)

        # If this is a final response (no tool calls), return
        if llm_output.is_final:
            return AgentWorkflowOutput(
                final_response=llm_output.message.content,
                messages=workflow_input.messages,
                iterations=iteration + 1,
                status="completed",
            ).to_dict()

        # Execute tool calls in parallel (ADK pattern with when_all)
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

        # Wait for all tool executions to complete
        tool_output_tasks = [t for _, t in tool_tasks]
        tool_outputs_data = yield when_all(tool_output_tasks)

        # Append tool result messages
        for i, tool_output_data in enumerate(tool_outputs_data):
            tool_output = ExecuteToolOutput.from_dict(tool_output_data)
            tool_result_message = Message(
                role=MessageRole.TOOL,
                content=str(tool_output.tool_result.result)
                if tool_output.tool_result.result
                else None,
                tool_call_id=tool_output.tool_result.tool_call_id,
                name=tool_output.tool_result.tool_name,
            )
            workflow_input.messages.append(tool_result_message)

        # Increment iteration counter
        iteration += 1

    # Max iterations reached
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
    """Activity that calls the OpenAI Chat Completions API.

    This activity uses openai.OpenAI().chat.completions.create() directly
    (not Runner.run()) to call the configured model with the conversation
    history and tool definitions.

    Args:
        ctx: The workflow activity context
        input_data: Dictionary containing CallLlmInput data

    Returns:
        CallLlmOutput as a dictionary
    """
    llm_input = CallLlmInput.from_dict(input_data)

    try:
        import openai

        # Build messages for the OpenAI API
        messages: list[dict[str, Any]] = []

        # Add system message from agent instructions
        if llm_input.agent_config.instructions:
            messages.append(
                {
                    "role": "system",
                    "content": _build_system_prompt(llm_input.agent_config),
                }
            )

        # Convert workflow messages to OpenAI Chat Completions format
        for msg in llm_input.messages:
            if msg.role == MessageRole.USER:
                messages.append(
                    {
                        "role": "user",
                        "content": msg.content or "",
                    }
                )
            elif msg.role == MessageRole.ASSISTANT:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                }
                if msg.content:
                    assistant_msg["content"] = msg.content
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.args),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(assistant_msg)
            elif msg.role == MessageRole.TOOL:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "",
                        "content": msg.content or "",
                    }
                )

        # Build tool definitions for the API
        tools: list[dict[str, Any]] = []
        for tool_def in llm_input.agent_config.tool_definitions:
            func_schema: dict[str, Any] = {
                "name": tool_def.name,
                "description": tool_def.description,
            }
            if tool_def.parameters:
                func_schema["parameters"] = tool_def.parameters
            else:
                func_schema["parameters"] = {
                    "type": "object",
                    "properties": {},
                }
            tools.append({"type": "function", "function": func_schema})

        # Call OpenAI Chat Completions API directly
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=llm_input.agent_config.model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools if tools else openai.NOT_GIVEN,  # type: ignore[arg-type]
        )

        # Parse response
        choice = response.choices[0]
        response_message = choice.message

        # Extract tool calls
        tool_calls = []
        if response_message.tool_calls:
            for tc in response_message.tool_calls:
                args = tc.function.arguments  # type: ignore[union-attr]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,  # type: ignore[union-attr]
                        args=args,
                    )
                )

        # Create response message
        output_message = Message(
            role=MessageRole.ASSISTANT,
            content=response_message.content,
            tool_calls=tool_calls,
        )

        # Determine if this is a final response (no tool calls)
        is_final = len(tool_calls) == 0

        return CallLlmOutput(
            message=output_message,
            is_final=is_final,
        ).to_dict()

    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        import traceback

        traceback.print_exc()
        return CallLlmOutput(
            message=Message(role=MessageRole.ASSISTANT),
            is_final=True,
            error=str(e),
        ).to_dict()


def _build_system_prompt(agent_config: AgentConfig) -> str:
    """Build the system prompt for the agent from its instructions."""
    return agent_config.instructions


def execute_tool_activity(
    ctx: WorkflowActivityContext, input_data: dict[str, Any]
) -> dict[str, Any]:
    """Activity that executes a single tool.

    This activity:
    1. Gets the tool from the registry
    2. Executes the tool with the provided arguments
    3. Returns the result

    Args:
        ctx: The workflow activity context
        input_data: Dictionary containing ExecuteToolInput data

    Returns:
        ExecuteToolOutput as a dictionary
    """
    tool_input = ExecuteToolInput.from_dict(input_data)
    tool_call = tool_input.tool_call

    # Get the tool from registry
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
        # Execute the tool based on its type
        result = _execute_tool(tool, tool_call.args)

        # Serialize result
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        elif hasattr(result, "to_dict"):
            result = result.to_dict()
        elif not isinstance(result, (str, int, float, bool, list, dict, type(None))):
            result = str(result)

        return ExecuteToolOutput(
            tool_result=ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
            ),
        ).to_dict()

    except Exception as e:
        logger.error(f"Error executing tool '{tool_call.name}': {e}")
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


def _execute_tool(tool: Any, args: dict[str, Any]) -> Any:
    """Execute a tool and return the result.

    Handles different tool types:
    - OpenAI Agents SDK FunctionTool (with on_invoke_tool callback)
    - Plain callables
    - Async callables
    """
    import asyncio

    # OpenAI Agents SDK FunctionTool: invoke via on_invoke_tool(ctx, json_input)
    if hasattr(tool, "on_invoke_tool"):
        result = tool.on_invoke_tool(None, json.dumps(args))
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(result)
            finally:
                loop.close()
        return result

    # Plain callable
    if callable(tool):
        result = tool(**args)
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(result)
            finally:
                loop.close()
        return result

    raise TypeError(
        f"Tool {tool} is not callable and has no on_invoke_tool attribute"
    )
