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

"""Dapr Workflow definitions for durable ADK agent execution."""

import asyncio
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
)

logger = logging.getLogger(__name__)


# Global tool registry - tools are registered by the runner
_tool_registry: dict[str, Any] = {}


def register_tool(name: str, tool: Any) -> None:
    """Register a tool for use by the execute_tool activity."""
    _tool_registry[name] = tool


def get_registered_tool(name: str) -> Optional[Any]:
    """Get a registered tool by name."""
    return _tool_registry.get(name)


def clear_tool_registry() -> None:
    """Clear all registered tools."""
    _tool_registry.clear()


def adk_agent_workflow(
    ctx: DaprWorkflowContext, input_data: dict[str, Any]
) -> Generator[Any, Any, Any]:
    """Dapr Workflow that orchestrates a Google ADK agent execution.

    This workflow:
    1. Calls the LLM to get the next action (as an activity)
    2. If the LLM returns tool calls, executes each tool (as separate activities)
    3. Loops back until the LLM returns a final response

    All iterations run within a single workflow instance.

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

    iteration = 0

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

        # Execute each tool as a separate activity (preserves ADK features per-tool)
        tool_tasks = []
        for tool_call in llm_output.message.tool_calls:
            tool_input = ExecuteToolInput(
                tool_call=tool_call,
                agent_name=workflow_input.agent_config.name,
                session_id=workflow_input.session_id,
                user_id=workflow_input.user_id,
                app_name=workflow_input.app_name,
            )
            task = ctx.call_activity(
                execute_tool_activity,
                input=tool_input.to_dict(),
                retry_policy=retry_policy,
            )
            tool_tasks.append(task)

        # Wait for all tool executions to complete
        tool_outputs_data = yield when_all(tool_tasks)

        # Collect tool results into a message
        tool_results = []
        for tool_output_data in tool_outputs_data:
            tool_output = ExecuteToolOutput.from_dict(tool_output_data)
            tool_results.append(tool_output.tool_result)

        # Create tool results message
        tool_results_message = Message(
            role=MessageRole.USER,
            tool_results=tool_results,
        )
        workflow_input.messages.append(tool_results_message)

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
    """Activity that calls the LLM model using Google's genai client.

    Args:
        ctx: The workflow activity context
        input_data: Dictionary containing CallLlmInput data

    Returns:
        CallLlmOutput as a dictionary
    """
    try:
        from google.genai import Client
        from google.genai import types
    except ImportError as e:
        logger.error(f"Failed to import Google genai: {e}")
        return CallLlmOutput(
            message=Message(role=MessageRole.MODEL),
            is_final=True,
            error=f"Google genai not installed: {e}",
        ).to_dict()

    llm_input = CallLlmInput.from_dict(input_data)

    try:
        # Convert messages to genai Content format
        contents = []
        for msg in llm_input.messages:
            parts = []

            # Add text content
            if msg.content:
                parts.append(types.Part.from_text(text=msg.content))

            # Add tool calls (function calls from model)
            for tc in msg.tool_calls:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=tc.name,
                            args=tc.args,
                            id=tc.id,
                        )
                    )
                )

            # Add tool results (function responses to model)
            for tr in msg.tool_results:
                parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=tr.tool_name,
                            id=tr.tool_call_id,
                            response={"result": tr.result}
                            if tr.error is None
                            else {"error": tr.error},
                        )
                    )
                )

            if parts:
                role = "user" if msg.role == MessageRole.USER else "model"
                contents.append(types.Content(role=role, parts=parts))

        # Build tool declarations from registered tools
        tools: list[types.Tool] = []
        for tool_def in llm_input.agent_config.tool_definitions:
            tool = get_registered_tool(tool_def.name)
            if tool and hasattr(tool, "_get_declaration"):
                try:
                    declaration = tool._get_declaration()
                    if declaration:
                        # Find or create a Tool with function_declarations
                        if not tools:
                            tools.append(
                                types.Tool(function_declarations=[declaration])
                            )
                        else:
                            if tools[0].function_declarations is None:
                                tools[0].function_declarations = []
                            tools[0].function_declarations.append(declaration)
                except Exception as e:
                    logger.warning(
                        f"Failed to get declaration for tool {tool_def.name}: {e}"
                    )

        # Create generate config
        config = types.GenerateContentConfig(
            system_instruction=llm_input.agent_config.system_instruction,
            tools=tools if tools else None,  # type: ignore[arg-type]
        )

        # Call the LLM
        client = Client()
        response = client.models.generate_content(
            model=llm_input.agent_config.model,
            contents=contents,  # type: ignore[arg-type]
            config=config,
        )

        # Parse response
        if not response.candidates:
            return CallLlmOutput(
                message=Message(role=MessageRole.MODEL),
                is_final=True,
                error="No candidates in LLM response",
            ).to_dict()

        candidate = response.candidates[0]
        content = candidate.content

        # Extract tool calls and text from response
        tool_calls: list[ToolCall] = []
        text_parts = []

        if content and content.parts:
            for part in content.parts:
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=fc.id or f"call_{len(tool_calls)}",
                            name=fc.name or "",
                            args=dict(fc.args) if fc.args else {},
                        )
                    )
                elif part.text:
                    text_parts.append(part.text)

        response_message = Message(
            role=MessageRole.MODEL,
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
        )

        # Determine if this is a final response
        is_final = len(tool_calls) == 0

        return CallLlmOutput(
            message=response_message,
            is_final=is_final,
        ).to_dict()

    except Exception:
        logger.exception("Error calling LLM")
        raise


def execute_tool_activity(
    ctx: WorkflowActivityContext, input_data: dict[str, Any]
) -> dict[str, Any]:
    """Activity that executes a single ADK tool with full ADK context.

    This activity:
    1. Gets the tool from the registry
    2. Creates proper ADK context (Runner, Session, etc.)
    3. Executes the tool via ADK's normal flow
    4. Returns the result

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
        # Import ADK components
        from google.adk.tools.tool_context import ToolContext
        from google.adk.agents.invocation_context import InvocationContext
        from google.adk.sessions.in_memory_session_service import InMemorySessionService
        from google.adk.sessions.session import Session
        from google.adk.events.event_actions import EventActions
        from google.adk.agents.llm_agent import LlmAgent

        # Create session service and session
        session_service = InMemorySessionService()

        # Create or get session
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            session = loop.run_until_complete(
                session_service.create_session(
                    app_name=tool_input.app_name or "dapr_workflow",
                    user_id=tool_input.user_id or "workflow_user",
                    session_id=tool_input.session_id,
                )
            )
        finally:
            pass  # Keep loop open for tool execution

        # Create a minimal agent for context (tools need an agent reference)
        # This is a limitation - we create a dummy agent just for context
        dummy_agent = LlmAgent(
            name=tool_input.agent_name,
            model="gemini-2.0-flash",  # Not actually used
        )

        # Create invocation context
        from google.adk.agents.invocation_context import new_invocation_context_id

        invocation_context = InvocationContext(
            invocation_id=new_invocation_context_id(),
            session=session,
            session_service=session_service,
            agent=dummy_agent,
        )

        # Create tool context
        event_actions = EventActions()
        tool_context = ToolContext(
            invocation_context=invocation_context,
            function_call_id=tool_call.id,
            event_actions=event_actions,
        )

        # Execute the tool
        try:
            result = loop.run_until_complete(
                tool.run_async(args=tool_call.args, tool_context=tool_context)
            )
        finally:
            loop.close()

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
            )
        ).to_dict()

    except Exception:
        logger.exception("Error executing tool '%s'", tool_call.name)
        raise
