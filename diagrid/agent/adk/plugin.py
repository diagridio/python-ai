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

"""Dapr Workflow Plugin for Google ADK.

This plugin integrates with ADK's plugin system to intercept tool execution
and enable durable execution via Dapr Workflows.
"""

import logging
from typing import Any, Optional

from google.adk.plugins import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)


class PendingToolExecution(Exception):
    """Raised when a tool execution should be deferred to the workflow."""

    def __init__(
        self, tool_name: str, tool_args: dict[str, Any], function_call_id: str
    ):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.function_call_id = function_call_id
        super().__init__(f"Tool '{tool_name}' execution pending")


class DaprWorkflowPlugin(BasePlugin):
    """ADK Plugin that enables durable tool execution via Dapr Workflows.

    This plugin intercepts tool execution callbacks:
    - In "capture" mode: Captures tool calls and raises PendingToolExecution
    - In "inject" mode: Returns pre-computed results for tool calls

    The workflow uses this plugin to:
    1. Run ADK until it tries to execute tools (capture mode)
    2. Execute each tool as a Dapr activity
    3. Resume ADK with the tool results (inject mode)
    """

    def __init__(self):
        super().__init__(name="dapr_workflow")
        self._mode = "capture"  # "capture" or "inject"
        self._captured_tool_calls: list[dict[str, Any]] = []
        self._tool_results: dict[str, Any] = {}  # function_call_id -> result

    def set_capture_mode(self) -> None:
        """Set plugin to capture mode - captures tool calls."""
        self._mode = "capture"
        self._captured_tool_calls = []

    def set_inject_mode(self, tool_results: dict[str, Any]) -> None:
        """Set plugin to inject mode - returns pre-computed results.

        Args:
            tool_results: Dict mapping function_call_id to result
        """
        self._mode = "inject"
        self._tool_results = tool_results

    def get_captured_tool_calls(self) -> list[dict[str, Any]]:
        """Get the list of captured tool calls."""
        return self._captured_tool_calls.copy()

    def clear_captured_tool_calls(self) -> None:
        """Clear the captured tool calls."""
        self._captured_tool_calls = []

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        """Called before each tool execution.

        In capture mode: Records the tool call and returns a marker to skip execution.
        In inject mode: Returns the pre-computed result if available.

        Args:
            tool: The tool being executed
            tool_args: Arguments for the tool
            tool_context: The tool context

        Returns:
            Tool result dict to skip execution, or None to proceed
        """
        function_call_id = (
            tool_context.function_call_id or f"call_{len(self._captured_tool_calls)}"
        )

        if self._mode == "inject":
            # Return pre-computed result if available
            if function_call_id in self._tool_results:
                logger.debug(
                    f"Injecting result for tool '{tool.name}' (id={function_call_id})"
                )
                return self._tool_results[function_call_id]
            else:
                logger.warning(
                    f"No result found for tool '{tool.name}' (id={function_call_id})"
                )
                return None

        elif self._mode == "capture":
            # Capture the tool call
            tool_call_info = {
                "function_call_id": function_call_id,
                "tool_name": tool.name,
                "tool_args": tool_args,
            }
            self._captured_tool_calls.append(tool_call_info)
            logger.debug(f"Captured tool call: {tool.name} (id={function_call_id})")

            # Return None to let ADK execute the tool normally
            # The workflow will handle durability by wrapping the entire
            # agent iteration as an activity
            return None

        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        """Called after each tool execution.

        Args:
            tool: The tool that was executed
            tool_args: Arguments that were passed to the tool
            tool_context: The tool context
            result: The result from tool execution

        Returns:
            Modified result or None to keep original
        """
        # We don't modify results, just let them through
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict]:
        """Called when a tool execution fails.

        Args:
            tool: The tool that failed
            tool_args: Arguments that were passed to the tool
            tool_context: The tool context
            error: The exception that was raised

        Returns:
            Error recovery result or None to propagate error
        """
        logger.error(f"Tool '{tool.name}' failed: {error}")
        # Let the error propagate - the workflow will handle retries
        return None
