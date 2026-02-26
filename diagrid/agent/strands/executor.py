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

"""Dapr Workflow Tool Executor - Executes Strands tools as Dapr Workflow activities.

This module provides a custom ToolExecutor that dispatches tool calls to Dapr Workflow
activities, providing durability and replay capabilities.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from typing_extensions import override

if TYPE_CHECKING:
    from strands.agent import Agent
    from strands.telemetry.metrics import Trace
    from strands.tools.structured_output._structured_output_context import (
        StructuredOutputContext,
    )

from strands.tools.executors._executor import ToolExecutor
from strands.types._events import TypedEvent, ToolResultEvent, ToolStreamEvent
from strands.types.tools import ToolResult, ToolUse

logger = logging.getLogger(__name__)


class DaprWorkflowToolExecutor(ToolExecutor):
    """Tool executor that dispatches tool calls as Dapr Workflow activities.

    This executor intercepts tool calls from the Strands agent and executes them
    as Dapr Workflow activities, providing:

    - **Durability**: Tool execution state is checkpointed by Dapr
    - **Replay**: Failed workflows can resume from the last successful checkpoint
    - **Distributed execution**: Tools can run on different workers
    - **Observability**: Full tracing through Dapr's telemetry

    The executor can operate in two modes:
    1. **Workflow mode**: When running inside a Dapr workflow context, tool calls
       are dispatched as activities using `call_activity`.
    2. **Direct mode**: When no workflow context exists, tools execute directly
       (useful for testing or non-durable scenarios).

    Example:
        ```python
        from diagrid.agent.strands import DaprWorkflowToolExecutor
        from strands import Agent

        executor = DaprWorkflowToolExecutor(
            activity_prefix="agent_tool_",
        )

        agent = Agent(
            model="us.amazon.nova-pro-v1:0",
            tools=[search_tool, calculator_tool],
            tool_executor=executor,
        )
        ```

    Args:
        activity_prefix: Prefix for activity names (default: "strands_tool_")
        concurrent: Whether to execute multiple tool calls concurrently (default: True)
        workflow_context_var: Context variable name for workflow context (default: None)
    """

    def __init__(
        self,
        activity_prefix: str = "strands_tool_",
        concurrent: bool = True,
        workflow_context_var: str | None = None,
    ) -> None:
        """Initialize the Dapr Workflow Tool Executor.

        Args:
            activity_prefix: Prefix for Dapr activity names
            concurrent: Execute multiple tools concurrently when possible
            workflow_context_var: Name of context variable holding workflow context
        """
        self.activity_prefix = activity_prefix
        self.concurrent = concurrent
        self.workflow_context_var = workflow_context_var
        self._workflow_context: Any = None

    def set_workflow_context(self, ctx: Any) -> None:
        """Set the Dapr workflow context for activity dispatch.

        This is called by DaprAgentWorkflow when the agent runs inside a workflow.

        Args:
            ctx: The Dapr WorkflowActivityContext or WorkflowContext
        """
        self._workflow_context = ctx

    def clear_workflow_context(self) -> None:
        """Clear the workflow context (e.g., after workflow completes)."""
        self._workflow_context = None

    @property
    def is_workflow_mode(self) -> bool:
        """Check if we're running inside a Dapr workflow context."""
        return self._workflow_context is not None

    def _get_activity_name(self, tool_name: str) -> str:
        """Generate the Dapr activity name for a tool.

        Args:
            tool_name: The Strands tool name

        Returns:
            The Dapr activity name (prefix + tool_name)
        """
        return f"{self.activity_prefix}{tool_name}"

    async def _execute_tool_as_activity(
        self,
        agent: "Agent",
        tool_use: ToolUse,
        tool_results: list[ToolResult],
        cycle_trace: "Trace",
        cycle_span: Any,
        invocation_state: dict[str, Any],
        structured_output_context: "StructuredOutputContext | None" = None,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute a single tool as a Dapr workflow activity.

        Args:
            agent: The Strands agent
            tool_use: The tool invocation request
            tool_results: List to append results to
            cycle_trace: Trace for metrics
            cycle_span: Span for tracing
            invocation_state: Agent invocation state
            structured_output_context: Structured output context

        Yields:
            Tool execution events
        """
        tool_name = tool_use["name"]
        activity_name = self._get_activity_name(tool_name)

        if self.is_workflow_mode:
            # Execute as Dapr activity
            logger.debug(
                "tool=%s activity=%s | executing tool as Dapr activity",
                tool_name,
                activity_name,
            )

            # Prepare activity input - serialize tool_use and minimal context
            activity_input = {
                "tool_use": tool_use,
                "tool_name": tool_name,
                # Include serializable parts of invocation state
                "invocation_state_keys": list(invocation_state.keys()),
            }

            try:
                # Call the Dapr activity
                result = await self._workflow_context.call_activity(
                    activity_name,
                    input=activity_input,
                )

                # Convert activity result back to ToolResult
                tool_result: ToolResult = {
                    "toolUseId": str(tool_use.get("toolUseId")),
                    "status": result.get("status", "success"),
                    "content": result.get(
                        "content", [{"text": str(result.get("result", ""))}]
                    ),
                }

                yield ToolResultEvent(tool_result)
                tool_results.append(tool_result)

            except Exception as e:
                logger.exception(
                    "tool=%s activity=%s | Dapr activity execution failed",
                    tool_name,
                    activity_name,
                )
                error_result: ToolResult = {
                    "toolUseId": str(tool_use.get("toolUseId")),
                    "status": "error",
                    "content": [{"text": f"Dapr activity error: {str(e)}"}],
                }
                yield ToolResultEvent(error_result)
                tool_results.append(error_result)

        else:
            # Direct execution mode - fall back to standard executor behavior
            logger.debug(
                "tool=%s | no workflow context, executing directly",
                tool_name,
            )
            async for event in ToolExecutor._stream_with_trace(
                agent,
                tool_use,
                tool_results,
                cycle_trace,
                cycle_span,
                invocation_state,
                structured_output_context,
            ):
                yield event

    @override
    async def _execute(
        self,
        agent: "Agent",
        tool_uses: list[ToolUse],
        tool_results: list[ToolResult],
        cycle_trace: "Trace",
        cycle_span: Any,
        invocation_state: dict[str, Any],
        structured_output_context: "StructuredOutputContext | None" = None,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute tools according to Dapr workflow activity strategy.

        When running in workflow mode, tools are dispatched as Dapr activities.
        When not in workflow mode, falls back to direct execution.

        Args:
            agent: The Strands agent
            tool_uses: List of tool invocation requests
            tool_results: List to append results to
            cycle_trace: Trace for metrics
            cycle_span: Span for tracing
            invocation_state: Agent invocation state
            structured_output_context: Structured output context

        Yields:
            Tool execution events
        """
        if not self.is_workflow_mode:
            # Fall back to standard concurrent execution
            logger.debug("No workflow context - using standard execution")
            async for event in self._execute_direct(
                agent,
                tool_uses,
                tool_results,
                cycle_trace,
                cycle_span,
                invocation_state,
                structured_output_context,
            ):
                yield event
            return

        if self.concurrent and len(tool_uses) > 1:
            # Concurrent execution as parallel activities
            async for event in self._execute_concurrent_activities(
                agent,
                tool_uses,
                tool_results,
                cycle_trace,
                cycle_span,
                invocation_state,
                structured_output_context,
            ):
                yield event
        else:
            # Sequential execution
            for tool_use in tool_uses:
                async for event in self._execute_tool_as_activity(
                    agent,
                    tool_use,
                    tool_results,
                    cycle_trace,
                    cycle_span,
                    invocation_state,
                    structured_output_context,
                ):
                    yield event

    async def _execute_concurrent_activities(
        self,
        agent: "Agent",
        tool_uses: list[ToolUse],
        tool_results: list[ToolResult],
        cycle_trace: "Trace",
        cycle_span: Any,
        invocation_state: dict[str, Any],
        structured_output_context: "StructuredOutputContext | None" = None,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute multiple tools concurrently as Dapr activities.

        Uses asyncio task queue pattern similar to ConcurrentToolExecutor
        but dispatches to Dapr activities.

        Args:
            agent: The Strands agent
            tool_uses: List of tool invocation requests
            tool_results: List to append results to
            cycle_trace: Trace for metrics
            cycle_span: Span for tracing
            invocation_state: Agent invocation state
            structured_output_context: Structured output context

        Yields:
            Tool execution events from all tools
        """
        task_queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
        task_events = [asyncio.Event() for _ in tool_uses]
        stop_event = object()

        async def execute_task(
            task_id: int,
            tool_use: ToolUse,
        ) -> None:
            """Execute a single tool and put results in queue."""
            try:
                async for event in self._execute_tool_as_activity(
                    agent,
                    tool_use,
                    tool_results,
                    cycle_trace,
                    cycle_span,
                    invocation_state,
                    structured_output_context,
                ):
                    task_queue.put_nowait((task_id, event))
                    await task_events[task_id].wait()
                    task_events[task_id].clear()
            finally:
                task_queue.put_nowait((task_id, stop_event))

        # Start all tasks
        tasks = [
            asyncio.create_task(execute_task(task_id, tool_use))
            for task_id, tool_use in enumerate(tool_uses)
        ]

        # Yield events as they arrive
        task_count = len(tasks)
        while task_count:
            task_id, event = await task_queue.get()
            if event is stop_event:
                task_count -= 1
                continue

            yield event
            task_events[task_id].set()

    async def _execute_direct(
        self,
        agent: "Agent",
        tool_uses: list[ToolUse],
        tool_results: list[ToolResult],
        cycle_trace: "Trace",
        cycle_span: Any,
        invocation_state: dict[str, Any],
        structured_output_context: "StructuredOutputContext | None" = None,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute tools directly without Dapr (fallback mode).

        This is used when no workflow context is available, providing
        the same behavior as the standard ConcurrentToolExecutor.

        Args:
            agent: The Strands agent
            tool_uses: List of tool invocation requests
            tool_results: List to append results to
            cycle_trace: Trace for metrics
            cycle_span: Span for tracing
            invocation_state: Agent invocation state
            structured_output_context: Structured output context

        Yields:
            Tool execution events
        """
        if self.concurrent and len(tool_uses) > 1:
            # Concurrent execution using task queue pattern
            task_queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
            task_events = [asyncio.Event() for _ in tool_uses]
            stop_event = object()

            async def execute_task(task_id: int, tool_use: ToolUse) -> None:
                try:
                    async for event in ToolExecutor._stream_with_trace(
                        agent,
                        tool_use,
                        tool_results,
                        cycle_trace,
                        cycle_span,
                        invocation_state,
                        structured_output_context,
                    ):
                        task_queue.put_nowait((task_id, event))
                        await task_events[task_id].wait()
                        task_events[task_id].clear()
                finally:
                    task_queue.put_nowait((task_id, stop_event))

            tasks = [
                asyncio.create_task(execute_task(task_id, tool_use))
                for task_id, tool_use in enumerate(tool_uses)
            ]

            task_count = len(tasks)
            while task_count:
                task_id, event = await task_queue.get()
                if event is stop_event:
                    task_count -= 1
                    continue

                yield event
                task_events[task_id].set()
        else:
            # Sequential execution
            for tool_use in tool_uses:
                async for event in ToolExecutor._stream_with_trace(
                    agent,
                    tool_use,
                    tool_results,
                    cycle_trace,
                    cycle_span,
                    invocation_state,
                    structured_output_context,
                ):
                    yield event
