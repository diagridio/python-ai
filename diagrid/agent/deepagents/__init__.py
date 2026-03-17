# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Diagrid Agent Deep Agents - Durable execution of LangChain Deep Agents using Dapr Workflows.

This extension enables durable execution of Deep Agents (built on LangGraph)
using Dapr Workflows. Since Deep Agents compile to standard LangGraph
CompiledStateGraphs, this module wraps the existing LangGraph runner with
a convenience API tailored for the Deep Agents harness.

Example:
    ```python
    from deepagents import create_deep_agent
    from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner

    agent = create_deep_agent(
        model="openai:gpt-4o-mini",
        tools=[my_tool],
    )

    runner = DaprWorkflowDeepAgentRunner(agent=agent, name="my-agent")
    runner.start()

    async for event in runner.run_async(
        input={"messages": [{"role": "user", "content": "Hello"}]},
        thread_id="thread-1",
    ):
        print(event)

    runner.shutdown()
    ```
"""

from diagrid.agent.deepagents.runner import DaprWorkflowDeepAgentRunner
from diagrid.agent.deepagents.version import __version__

__all__ = [
    "DaprWorkflowDeepAgentRunner",
    "__version__",
]
