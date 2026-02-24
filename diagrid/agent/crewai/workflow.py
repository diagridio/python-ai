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

"""Dapr Workflow definitions for durable CrewAI agent execution."""

import json
import logging
from datetime import timedelta
from typing import Any, Callable, Generator, Optional

from dapr.ext.workflow import (
    DaprWorkflowContext,
    WorkflowActivityContext,
    RetryPolicy,
)

from diagrid.agent.core.chat import (
    ChatMessage,
    ChatRole,
    ChatToolCall,
    ChatToolDefinition,
    get_chat_client,
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
    TaskConfig,
    ToolCall,
    ToolResult,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


# Global tool registry - tools are registered by the runner
_tool_registry: dict[str, Any] = {}
_tool_definitions: dict[str, ToolDefinition] = {}
_default_workflow_input_factory: Optional[Callable[[str], dict[str, Any]]] = None


def register_tool(
    name: str, tool: Any, definition: Optional[ToolDefinition] = None
) -> None:
    """Register a tool for use by the execute_tool activity.

    Args:
        name: The tool name
        tool: The actual tool object (CrewAI BaseTool or callable)
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


def set_default_workflow_input_factory(
    factory: Callable[[str], dict[str, Any]],
) -> None:
    """Store a factory that builds AgentWorkflowInput dicts from a task string."""
    global _default_workflow_input_factory
    _default_workflow_input_factory = factory


def clear_tool_registry() -> None:
    """Clear all registered tools."""
    global _default_workflow_input_factory
    _tool_registry.clear()
    _tool_definitions.clear()
    _default_workflow_input_factory = None


def agent_workflow(
    ctx: DaprWorkflowContext, input_data: dict[str, Any]
) -> Generator[Any, Any, Any]:
    """Dapr Workflow that orchestrates a CrewAI agent execution.

    This workflow:
    1. Calls the LLM to get the next action (as an activity)
    2. If the LLM returns tool calls, executes each tool (as separate activities)
    3. Loops back until the LLM returns a final response

    All iterations run within a single workflow instance, making the entire
    agent execution durable and resumable.

    Args:
        ctx: The Dapr workflow context
        input_data: Dictionary containing AgentWorkflowInput data

    Returns:
        AgentWorkflowOutput as a dictionary
    """
    # Deserialize input - handle orchestrator calls with {"task": "..."}
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
            task_config=workflow_input.task_config,
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

        # Execute each tool call sequentially (CrewAI executes one at a time
        # to allow the LLM to reflect on each result)
        for tool_call in llm_output.message.tool_calls:
            tool_input = ExecuteToolInput(
                tool_call=tool_call,
                agent_role=workflow_input.agent_config.role,
                session_id=workflow_input.session_id,
            )

            tool_output_data = yield ctx.call_activity(
                execute_tool_activity,
                input=tool_input.to_dict(),
                retry_policy=retry_policy,
            )
            tool_output = ExecuteToolOutput.from_dict(tool_output_data)

            # Create tool result message — include error text so the
            # LLM can see what went wrong and self-correct.
            tr = tool_output.tool_result
            if tr.error:
                content = f"Error: {tr.error}"
            elif tr.result is not None:
                content = str(tr.result)
            else:
                content = ""
            tool_result_message = Message(
                role=MessageRole.TOOL,
                content=content,
                tool_call_id=tr.tool_call_id,
                name=tr.tool_name,
            )
            workflow_input.messages.append(tool_result_message)

            # If tool has result_as_answer, return immediately
            if tool_output.result_as_answer:
                return AgentWorkflowOutput(
                    final_response=str(tool_output.tool_result.result),
                    messages=workflow_input.messages,
                    iterations=iteration + 1,
                    status="completed",
                ).to_dict()

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


def _call_llm_via_dapr(llm_input: CallLlmInput) -> dict[str, Any]:
    """Route LLM call through Dapr Conversation API."""
    # Build system prompt
    system_content = _build_system_prompt(llm_input.agent_config, llm_input.task_config)

    chat_messages = [ChatMessage(role=ChatRole.SYSTEM, content=system_content)]

    for msg in llm_input.messages:
        if msg.role == MessageRole.USER:
            chat_messages.append(
                ChatMessage(role=ChatRole.USER, content=msg.content or "")
            )
        elif msg.role == MessageRole.ASSISTANT:
            tool_calls = [
                ChatToolCall(id=tc.id, name=tc.name, arguments=json.dumps(tc.args))
                for tc in msg.tool_calls
            ]
            chat_messages.append(
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content=msg.content,
                    tool_calls=tool_calls,
                )
            )
        elif msg.role == MessageRole.TOOL:
            chat_messages.append(
                ChatMessage(
                    role=ChatRole.TOOL,
                    content=msg.content or "",
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            )

    # Convert tool definitions
    tools = [
        ChatToolDefinition(
            name=td.name,
            description=td.description,
            parameters=td.parameters or {"type": "object", "properties": {}},
        )
        for td in llm_input.agent_config.tool_definitions
    ] or None

    client = get_chat_client(llm_input.agent_config.component_name)
    response = client.chat(messages=chat_messages, tools=tools)

    # Convert ChatResponse -> CallLlmOutput
    fw_tool_calls: list[ToolCall] = []
    for tc in response.tool_calls:
        try:
            args = json.loads(tc.arguments)
        except (json.JSONDecodeError, TypeError):
            args = {}
        fw_tool_calls.append(ToolCall(id=tc.id, name=tc.name, args=args))

    output_message = Message(
        role=MessageRole.ASSISTANT,
        content=response.content,
        tool_calls=fw_tool_calls,
    )

    return CallLlmOutput(
        message=output_message,
        is_final=response.is_final,
    ).to_dict()


def call_llm_activity(
    ctx: WorkflowActivityContext, input_data: dict[str, Any]
) -> dict[str, Any]:
    """Activity that calls the LLM model to decide the next action.

    This activity uses the LiteLLM library (via CrewAI's LLM class) to call
    the configured model with the conversation history and tool definitions.
    If a component_name is set, routes through the Dapr Conversation API instead.

    Args:
        ctx: The workflow activity context
        input_data: Dictionary containing CallLlmInput data

    Returns:
        CallLlmOutput as a dictionary
    """
    llm_input = CallLlmInput.from_dict(input_data)

    # Route through Dapr if component_name is set
    if llm_input.agent_config.component_name:
        return _call_llm_via_dapr(llm_input)

    try:
        # Import LiteLLM for model calls
        from litellm import completion

        # Build messages for the LLM
        messages = []

        # Add system message with agent context
        system_content = _build_system_prompt(
            llm_input.agent_config, llm_input.task_config
        )
        messages.append(
            {
                "role": "system",
                "content": system_content,
            }
        )

        # Convert workflow messages to LiteLLM format
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
                        "name": msg.name or "",
                        "content": msg.content or "",
                    }
                )

        # Build tool definitions for the LLM
        tools: list[dict[str, Any]] = []
        for tool_def in llm_input.agent_config.tool_definitions:
            func_schema: dict[str, Any] = {
                "name": tool_def.name,
                "description": tool_def.description,
            }
            if tool_def.parameters:
                func_schema["parameters"] = tool_def.parameters
            else:
                # Default to empty object schema if no parameters defined
                func_schema["parameters"] = {
                    "type": "object",
                    "properties": {},
                }
            tools.append({"type": "function", "function": func_schema})

        # Call the LLM
        from diagrid.agent.core.telemetry import get_tracer

        _tracer = get_tracer("crewai.agent")
        _span = _tracer.start_span("LLM.completion") if _tracer else None
        if _span:
            _span.set_attribute("llm.model", llm_input.agent_config.model)
        try:
            response = completion(
                model=llm_input.agent_config.model,
                messages=messages,
                tools=tools if tools else None,
            )
        except Exception:
            if _span:
                _span.set_attribute("error", True)
            raise
        finally:
            if _span:
                _span.end()

        # Parse response
        choice = response.choices[0]
        response_message = choice.message

        # Extract tool calls
        tool_calls = []
        if response_message.tool_calls:
            for tc in response_message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
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

    except Exception:
        logger.exception("Error calling LLM")
        raise


def _build_system_prompt(agent_config: AgentConfig, task_config: TaskConfig) -> str:
    """Build the system prompt for the agent."""
    # Use custom template if provided, otherwise build default
    if agent_config.system_template:
        return agent_config.system_template.format(
            role=agent_config.role,
            goal=agent_config.goal,
            backstory=agent_config.backstory,
            task_description=task_config.description,
            expected_output=task_config.expected_output,
        )

    return f"""You are {agent_config.role}.

Your Goal: {agent_config.goal}

Background: {agent_config.backstory}

Current Task:
{task_config.description}

Expected Output:
{task_config.expected_output}

{f"Additional Context: {task_config.context}" if task_config.context else ""}

Instructions:
- Use the available tools when needed to complete the task
- Think step by step about what you need to do
- When you have the final answer, provide it clearly without using tools
- Be concise but thorough in your responses"""


def execute_tool_activity(
    ctx: WorkflowActivityContext, input_data: dict[str, Any]
) -> dict[str, Any]:
    """Activity that executes a single CrewAI tool.

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
    tool_def = get_tool_definition(tool_call.name)

    if tool is None:
        return ExecuteToolOutput(
            tool_result=ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=None,
                error=f"Tool '{tool_call.name}' not found in registry",
            )
        ).to_dict()

    from diagrid.agent.core.telemetry import get_tracer

    _tracer = get_tracer("crewai.agent")
    _span = _tracer.start_span("Tool.execute") if _tracer else None
    if _span:
        _span.set_attribute("tool.name", tool_call.name)
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

        # Check if this tool's result should be the final answer
        result_as_answer = tool_def.result_as_answer if tool_def else False

        return ExecuteToolOutput(
            tool_result=ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
            ),
            result_as_answer=result_as_answer,
        ).to_dict()

    except Exception as e:
        logger.error(f"Error executing tool '{tool_call.name}': {e}")
        if _span:
            _span.set_attribute("error", True)
        return ExecuteToolOutput(
            tool_result=ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=None,
                error=str(e),
            ),
        ).to_dict()
    finally:
        if _span:
            _span.end()


def _execute_tool(tool: Any, args: dict[str, Any]) -> Any:
    """Execute a tool and return the result.

    Handles different tool types:
    - CrewAI BaseTool instances (with run() or _run() methods)
    - Callable functions
    - LangChain-style tools
    """
    import asyncio

    # Check if it's a CrewAI BaseTool
    if hasattr(tool, "_run"):
        # CrewAI BaseTool - call _run directly
        result = tool._run(**args)
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(result)
            finally:
                loop.close()
        return result

    elif hasattr(tool, "run"):
        # Tool with run method
        result = tool.run(**args)
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(result)
            finally:
                loop.close()
        return result

    elif hasattr(tool, "invoke"):
        # LangChain-style tool
        result = tool.invoke(args)
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(result)
            finally:
                loop.close()
        return result

    elif callable(tool):
        # Plain callable
        result = tool(**args)
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(result)
            finally:
                loop.close()
        return result

    else:
        raise TypeError(
            f"Tool {tool} is not callable and has no run/_run/invoke method"
        )
