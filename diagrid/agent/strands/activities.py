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

"""Dapr Tool Activities - Register Strands tools as Dapr workflow activities.

This module provides utilities for registering Strands tools as Dapr workflow
activities, enabling durable execution and replay.
"""

import logging
from dataclasses import dataclass
from typing import Any, TypeVar

from strands.types.tools import AgentTool, ToolUse, ToolResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class DaprToolActivity:
    """Represents a Strands tool registered as a Dapr activity.

    This class wraps a Strands tool and provides the interface needed
    for Dapr workflow activity execution.

    Attributes:
        tool: The underlying Strands AgentTool
        activity_name: The Dapr activity name
        retry_policy: Optional retry policy for the activity
    """

    tool: AgentTool
    activity_name: str
    retry_policy: Any | None = None

    @property
    def tool_name(self) -> str:
        """The Strands tool name."""
        return self.tool.tool_name

    async def execute(self, ctx: Any, input_data: dict) -> dict:  # type: ignore[type-arg]
        """Execute the tool as a Dapr activity.

        Args:
            ctx: The Dapr activity context
            input_data: The activity input containing tool_use data

        Returns:
            The tool result as a dict
        """
        tool_use: ToolUse = input_data.get("tool_use", {})

        logger.debug(
            "activity=%s tool=%s | executing tool activity",
            self.activity_name,
            self.tool_name,
        )

        try:
            # Execute the tool
            result_content: list[dict[str, Any]] = []

            async for event in self.tool.stream(tool_use, {}):
                # Collect results from the tool stream
                if isinstance(event, dict):
                    if "toolResult" in event:
                        return {
                            "status": event["toolResult"].get("status", "success"),
                            "content": event["toolResult"].get("content", []),
                        }
                    elif "text" in event:
                        result_content.append({"text": event["text"]})
                    elif "result" in event:
                        result_content.append({"text": str(event["result"])})

            # If we got content from streaming, return it
            if result_content:
                return {"status": "success", "content": result_content}

            # Handle final event
            if event is not None:
                return {
                    "status": "success",
                    "content": [{"text": str(event)}],
                }

            return {"status": "success", "content": [{"text": "Tool completed"}]}

        except Exception as e:
            logger.exception(
                "activity=%s tool=%s | tool execution failed",
                self.activity_name,
                self.tool_name,
            )
            return {
                "status": "error",
                "content": [{"text": f"Error: {str(e)}"}],
            }


def create_tool_activity(
    tool: AgentTool,
    activity_prefix: str = "strands_tool_",
    retry_policy: Any | None = None,
) -> DaprToolActivity:
    """Create a Dapr activity wrapper for a Strands tool.

    Args:
        tool: The Strands AgentTool to wrap
        activity_prefix: Prefix for the activity name
        retry_policy: Optional Dapr retry policy

    Returns:
        A DaprToolActivity instance
    """
    activity_name = f"{activity_prefix}{tool.tool_name}"

    return DaprToolActivity(
        tool=tool,
        activity_name=activity_name,
        retry_policy=retry_policy,
    )


def register_tool_activities(
    workflow_runtime: Any,
    tools: list[AgentTool],
    activity_prefix: str = "strands_tool_",
    retry_policy: Any | None = None,
) -> list[DaprToolActivity]:
    """Register multiple Strands tools as Dapr workflow activities.

    This function creates and registers Dapr activities for each tool,
    enabling them to be called from within workflows.

    Example:
        ```python
        from dapr.ext.workflow import WorkflowRuntime
        from diagrid.agent.strands import register_tool_activities
        from strands import tool

        @tool
        def search(query: str) -> str:
            '''Search the web.'''
            return f"Results for: {query}"

        @tool
        def calculate(expression: str) -> float:
            '''Evaluate a math expression.'''
            return eval(expression)

        runtime = WorkflowRuntime()
        activities = register_tool_activities(
            runtime,
            [search, calculate],
        )
        ```

    Args:
        workflow_runtime: The Dapr WorkflowRuntime instance
        tools: List of Strands tools to register
        activity_prefix: Prefix for activity names
        retry_policy: Optional default retry policy for all activities

    Returns:
        List of registered DaprToolActivity instances
    """
    registered: list[DaprToolActivity] = []

    for tool in tools:
        # Handle both decorated functions and AgentTool instances
        if hasattr(tool, "tool_name"):
            agent_tool = tool
        elif hasattr(tool, "__wrapped__"):
            # Decorated function - get the AgentTool wrapper
            agent_tool = tool
        else:
            logger.warning(
                "tool=%s | skipping - not a valid AgentTool",
                getattr(tool, "__name__", str(tool)),
            )
            continue

        activity = create_tool_activity(
            tool=agent_tool,
            activity_prefix=activity_prefix,
            retry_policy=retry_policy,
        )

        # Register with Dapr
        workflow_runtime.activity(name=activity.activity_name)(activity.execute)

        registered.append(activity)

        logger.debug(
            "tool=%s activity=%s | registered tool activity",
            activity.tool_name,
            activity.activity_name,
        )

    logger.info(
        "activities=%d | registered tool activities with Dapr",
        len(registered),
    )

    return registered


class ToolActivityRegistry:
    """Registry for managing tool activities.

    This class provides a centralized way to manage tool activities
    and their registration with Dapr.

    Example:
        ```python
        registry = ToolActivityRegistry(activity_prefix="my_agent_")

        # Add tools
        registry.add(search_tool)
        registry.add(calculate_tool)

        # Register all with Dapr
        registry.register_all(workflow_runtime)

        # Get activity by tool name
        activity = registry.get("search")
        ```
    """

    def __init__(
        self,
        activity_prefix: str = "strands_tool_",
        default_retry_policy: Any | None = None,
    ) -> None:
        """Initialize the registry.

        Args:
            activity_prefix: Prefix for activity names
            default_retry_policy: Default retry policy for activities
        """
        self.activity_prefix = activity_prefix
        self.default_retry_policy = default_retry_policy
        self._activities: dict[str, DaprToolActivity] = {}

    def add(
        self,
        tool: AgentTool,
        retry_policy: Any | None = None,
    ) -> DaprToolActivity:
        """Add a tool to the registry.

        Args:
            tool: The Strands AgentTool to add
            retry_policy: Optional retry policy (overrides default)

        Returns:
            The created DaprToolActivity
        """
        activity = create_tool_activity(
            tool=tool,
            activity_prefix=self.activity_prefix,
            retry_policy=retry_policy or self.default_retry_policy,
        )

        self._activities[activity.tool_name] = activity
        return activity

    def get(self, tool_name: str) -> DaprToolActivity | None:
        """Get an activity by tool name.

        Args:
            tool_name: The Strands tool name

        Returns:
            The DaprToolActivity or None if not found
        """
        return self._activities.get(tool_name)

    def get_activity_name(self, tool_name: str) -> str | None:
        """Get the Dapr activity name for a tool.

        Args:
            tool_name: The Strands tool name

        Returns:
            The Dapr activity name or None if not found
        """
        activity = self.get(tool_name)
        return activity.activity_name if activity else None

    @property
    def activities(self) -> list[DaprToolActivity]:
        """All registered activities."""
        return list(self._activities.values())

    @property
    def activity_names(self) -> list[str]:
        """All registered activity names."""
        return [a.activity_name for a in self._activities.values()]

    def register_all(self, workflow_runtime: Any) -> None:
        """Register all activities with a Dapr workflow runtime.

        Args:
            workflow_runtime: The Dapr WorkflowRuntime instance
        """
        for activity in self._activities.values():
            workflow_runtime.activity(name=activity.activity_name)(activity.execute)

        logger.info(
            "activities=%d | registered all tool activities",
            len(self._activities),
        )

    def clear(self) -> None:
        """Clear all registered activities."""
        self._activities.clear()
