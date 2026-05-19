# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Diagrid Agent Claude Agent SDK — Durable execution of Claude agents.

This extension wraps the Claude Agent SDK so each agent invocation runs as a
Dapr Workflow. Every LLM turn and every tool call become individual durable
activities under that workflow, so a crash mid-iteration resumes from the
last checkpoint instead of restarting the conversation.

Example:
    ```python
    from claude_agent_sdk import tool
    from diagrid.agent.claude_agents import DaprWorkflowAgentRunner

    @tool("get_weather", "Get the weather for a city", {"city": str})
    async def get_weather(args):
        return {"content": [{"type": "text", "text": f"Sunny in {args['city']}"}]}

    runner = DaprWorkflowAgentRunner(
        name="weather-agent",
        system_prompt="You are a weather assistant.",
        model="claude-sonnet-4-6",
        tools=[get_weather],
    )
    runner.start()

    async for event in runner.run_async(
        user_message="What's the weather in Tokyo?",
        session_id="session-123",
    ):
        print(event)

    runner.shutdown()
    ```

Benefits:
    - Fault tolerance: Workflow resumes from the last successful activity
      after a crash or restart.
    - Granularity: Each LLM call and each tool call is its own activity, so
      retries and checkpoints are per-step.
    - Observability: All steps are visible through Dapr's workflow APIs.

Install:
    pip install diagrid claude-agent-sdk anthropic
"""

from diagrid.agent.claude_agents.models import (
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
from diagrid.agent.claude_agents.runner import DaprWorkflowAgentRunner
from diagrid.agent.claude_agents.state import DaprMemoryStore
from diagrid.agent.claude_agents.version import __version__
from diagrid.agent.claude_agents.workflow import (
    agent_workflow,
    call_llm_activity,
    clear_tool_registry,
    execute_tool_activity,
    get_registered_tool,
    get_tool_definition,
    register_tool,
)

__all__ = [
    # Main runner class
    "DaprWorkflowAgentRunner",
    # State
    "DaprMemoryStore",
    # Data models
    "AgentConfig",
    "AgentWorkflowInput",
    "AgentWorkflowOutput",
    "CallLlmInput",
    "CallLlmOutput",
    "ExecuteToolInput",
    "ExecuteToolOutput",
    "Message",
    "MessageRole",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    # Workflow and activities (advanced)
    "agent_workflow",
    "call_llm_activity",
    "execute_tool_activity",
    # Tool registry (advanced)
    "register_tool",
    "get_registered_tool",
    "get_tool_definition",
    "clear_tool_registry",
    # Version
    "__version__",
]
